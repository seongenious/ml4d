"""
RT-1 변형 (자율주행; 인식토큰 없음)
입력: [B, T, K, 3, H, W]  (T=시계열 프레임 수, K=카메라 수)
출력: lateral(조향) 시퀀스 H=12, longitudinal(가/감속) 시퀀스 H=12 (각 스텝 256-way), blinker(2-way)
주의: 데이터셋에서 입력/라벨의 타임스탬프 정렬은 0.5s 간격으로 이미 처리되어 있다고 가정.
"""

import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------ FiLM ------------------------
class FiLM(nn.Module):
    def __init__(self, feature_dim: int, prompt_dim: int):
        super().__init__()
        self.film_proj = nn.Sequential(
            nn.Linear(prompt_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim * 2),
        )

    def forward(self, features: torch.Tensor, prompt_embed: torch.Tensor) -> torch.Tensor:
        # features: [B, C, H, W], prompt_embed: [B, D]
        gamma_beta = self.film_proj(prompt_embed)  # [B, 2C]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return features * (1.0 + gamma) + beta


# --------------------- TokenLearner -------------------
class TokenLearner(nn.Module):
    """공간 피처 -> m개의 토큰 (정규화된 가중합)"""
    def __init__(self, in_ch: int, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.selector = nn.Sequential(
            nn.Conv2d(in_ch, max(in_ch // 4, 16), 1),
            nn.GELU(),
            nn.Conv2d(max(in_ch // 4, 16), num_tokens, 1),
        )
        self.proj = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        attn = self.selector(x)  # [B, m, H, W]
        attn = attn.flatten(2)   # [B, m, HW]
        attn = attn.softmax(-1)  # 정규화
        x_proj = self.proj(x).flatten(2)  # [B, C, HW]
        # 가중합: 토큰 i = Σ_hw attn[i, hw] * feat[:, :, hw]
        tokens = torch.einsum("bmh,bch->bmc", attn, x_proj)  # [B, m, C]
        return tokens


# -------------------- Vision Backbone -----------------
class EfficientNetBackbone(nn.Module):
    """timm EfficientNet-B1 (features_only, pretrained) + FiLM on last feature"""
    def __init__(self, prompt_dim: int = 512, model_name: str = "tf_efficientnet_b1"):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise RuntimeError("timm이 필요합니다: pip install timm") from e

        self.backbone = timm.create_model(
            model_name, features_only=True, pretrained=True
        )
        self.out_ch = self.backbone.feature_info.channels()[-1]
        self.film = FiLM(self.out_ch, prompt_dim)

    def forward(self, x: torch.Tensor, prompt_embed: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W]
        feats: List[torch.Tensor] = self.backbone(x)  # list of stages
        f = feats[-1]  # [B, C, H', W']
        f = self.film(f, prompt_embed)  # FiLM modulation
        return f  # [B, C, H', W']


# -------------------- Prompt Encoder ------------------
class PromptEncoder(nn.Module):
    """
    우선 HF 사전학습 텍스트 인코더 사용을 시도하고,
    불가 시 (오프라인 등) 간단 LSTM으로 fallback.
    """
    def __init__(self, embed_dim: int = 512, hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_hf = False
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
            self.text_model = AutoModel.from_pretrained(hf_model)
            self.proj = nn.Linear(self.text_model.config.hidden_size, embed_dim)
            self.use_hf = True
        except Exception:
            # fallback
            vocab_size = 10000
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True)
            self.proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, prompt) -> torch.Tensor:
        """
        prompt: List[str] 또는 텐서(토큰). HF 사용 시 List[str] 권장.
        return: [B, embed_dim]
        """
        if self.use_hf:
            toks = self.tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
            toks = {k: v.to(next(self.parameters()).device) for k, v in toks.items()}
            out = self.text_model(**toks).last_hidden_state  # [B, L, H]
            pooled = out[:, 0]  # CLS
            return self.proj(pooled)
        else:
            # prompt: LongTensor [B, L]
            emb = self.embedding(prompt)
            lstm_out, _ = self.lstm(emb)
            return self.proj(lstm_out[:, -1, :])


# --------------------- Action Heads -------------------
class ActionHeads(nn.Module):
    """
    비-AR 시퀀스 헤드:
    입력: policy_repr [B, D]
    출력:
      - lateral: [B, H, 256]
      - longitudinal: [B, H, 256]
      - blinker: [B, 2]
    """
    def __init__(self, d_model: int, horizon: int = 12, bins: int = 256):
        super().__init__()
        self.h = horizon
        self.bins = bins
        self.lat_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, horizon * bins)
        )
        self.lon_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, horizon * bins)
        )
        self.blinker = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, policy_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, D = policy_repr.shape
        lat = self.lat_head(policy_repr).view(B, self.h, self.bins)
        lon = self.lon_head(policy_repr).view(B, self.h, self.bins)
        bln = self.blinker(policy_repr)  # [B,2]
        return {"lateral": lat, "longitudinal": lon, "blinker": bln}


# ----------------------- Model ------------------------
class RT1AV(nn.Module):
    def __init__(
        self,
        num_cameras: int = 6,
        seq_len: int = 6,          # 입력 프레임 수 T
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        tokens_per_frame: int = 2, # TokenLearner 토큰 수(프레임당)
        horizon: int = 12,         # 액션 시퀀스 길이 H
        bins: int = 256,           # 액션 bins (조향/가감속)
        use_hf_text: bool = True
    ):
        super().__init__()
        self.K = num_cameras
        self.T = seq_len
        self.d_model = d_model

        # 텍스트 인코더 (사전학습 모델 우선)
        self.prompt_encoder = PromptEncoder(embed_dim=d_model) if use_hf_text else PromptEncoder(embed_dim=d_model, hf_model="__fallback__")

        # 하나의 공유 비전 백본 (pretrained) + 카메라 임베딩
        self.backbone = EfficientNetBackbone(prompt_dim=d_model)
        self.tokenlearner = TokenLearner(self.backbone.out_ch, tokens_per_frame)
        self.cam_embed = nn.Embedding(num_embeddings=num_cameras, embedding_dim=d_model)

        # 피처 -> d_model 정렬
        self.token_proj = nn.Linear(self.backbone.out_ch, d_model)

        # 정책용 [CLS] 토큰
        self.policy_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 위치 임베딩(최대 토큰 수 여유 있게)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, d_model))

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # 시퀀스 액션 헤드
        self.heads = ActionHeads(d_model=d_model, horizon=horizon, bins=bins)

    def forward(self, images: torch.Tensor, prompts) -> Dict[str, torch.Tensor]:
        """
        images: [B, T, K, 3, H, W]
        prompts: List[str] (HF) 또는 LongTensor [B, L] (fallback)
        return: dict with logits:
            lateral:      [B, H, 256]
            longitudinal: [B, H, 256]
            blinker:      [B, 2]
        """
        B, T, K, C, H, W = images.shape
        assert T == self.T and K == self.K, f"입력 차원 불일치: got T={T},K={K}, expected T={self.T},K={self.K}"

        # 텍스트 임베딩
        prompt_embed = self.prompt_encoder(prompts)  # [B, D]

        # 프레임×카메라 순회: (FiLM 백본 → TokenLearner → proj → cam_embed 더하기)
        tokens_list = []
        for t in range(T):
            for k in range(K):
                frame = images[:, t, k]  # [B,3,H,W]
                feat = self.backbone(frame, prompt_embed)           # [B, C', h, w]
                tok  = self.tokenlearner(feat)                      # [B, m, C']
                tok  = self.token_proj(tok)                         # [B, m, D]
                cam_e = self.cam_embed.weight[k].view(1, 1, -1)     # [1,1,D]
                tok = tok + cam_e                                   # 카메라 임베딩 주입
                tokens_list.append(tok)

        # 시퀀스 구성
        tokens = torch.cat(tokens_list, dim=1)  # [B, T*K*m, D]
        # 정책 CLS 토큰 prepend
        cls = self.policy_token.expand(B, -1, -1)                  # [B,1,D]
        seq = torch.cat([cls, tokens], dim=1)                      # [B, 1+N, D]

        # 위치 임베딩
        pos = self.pos_embed[:, :seq.size(1), :]                   # [1, 1+N, D]
        seq = seq + pos

        # Transformer 인코딩
        enc = self.encoder(seq)                                    # [B, 1+N, D]

        # 정책 표현 = CLS 토큰 출력
        policy_repr = enc[:, 0, :]                                 # [B, D]

        # 액션 시퀀스 로짓
        logits = self.heads(policy_repr)                           # dict
        return logits

    @staticmethod
    def loss_fn(logits: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        targets:
          - lateral_bin:      [B, H]  (0..255)
          - longitudinal_bin: [B, H]  (0..255)
          - blinker:          [B]     (0/1)
        """
        B, H, V = logits["lateral"].shape
        loss_lat = F.cross_entropy(logits["lateral"].view(B * H, V), targets["lateral_bin"].view(B * H))
        loss_lon = F.cross_entropy(logits["longitudinal"].view(B * H, V), targets["longitudinal_bin"].view(B * H))
        loss_bln = F.cross_entropy(logits["blinker"], targets["blinker"])
        loss = loss_lat + loss_lon + 0.1 * loss_bln
        return {"loss": loss, "loss_lat": loss_lat, "loss_lon": loss_lon, "loss_bln": loss_bln}


# -------------------- Factory -------------------------
def create_rt1_model(config: Dict) -> RT1AV:
    return RT1AV(
        num_cameras=len(config["data"]["cameras"]),
        seq_len=config["data"]["seq_len"],          # T
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["transformer_layers"],
        tokens_per_frame=config["model"]["tokenlearner_tokens_per_frame"],
        horizon=12,                                  # 고정 (dt_out=0.5s × 12)
        bins=256,
        use_hf_text=True,
    )


# --------------------- Quick Test ---------------------
if __name__ == "__main__":
    B, T, K = 2, 6, 6
    x = torch.randn(B, T, K, 3, 224, 224)
    prompts = ["직진 유지"] * B  # HF 경로 (오프라인이면 PromptEncoder가 자동 fallback)
    model = RT1AV(num_cameras=K, seq_len=T)
    with torch.no_grad():
        y = model(x, prompts)
    print({k: v.shape for k, v in y.items()})
    # 기대: {'lateral': [B,12,256], 'longitudinal':[B,12,256], 'blinker':[B,2]}
