# src/train.py (요지)
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger

mlf_logger = MLFlowLogger(
    experiment_name="rt1_av_lane_keep_poc",
    tracking_uri="file:/workspace/mlruns"  # UI와 동일한 루트
)

trainer = Trainer(
    max_epochs=20,
    precision="16-mixed",      # AMP
    logger=mlf_logger,
    default_root_dir="/workspace/logs",
    devices="auto",
    accelerator="gpu"
)

# trainer.fit(model, datamodule=...)
