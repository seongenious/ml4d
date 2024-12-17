# **ML4D: Machine Learning for Autonomous Driving**

**ML4D** is a project that integrates machine learning-based simulators, data generators, and motion planners for autonomous driving.

---

## **Project Setup and Execution**

### **Prerequisites**

Ensure the following are installed:  

- **Docker**: [Installation Guide](https://docs.docker.com/engine/install/)  
- **NVIDIA Container Toolkit** (for GPU support): [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)  

---

### **Start with Docker Image**

To build the Docker image, run:

```bash
./docker.sh build
```

To start the container and launch Jupyter Notebook:

```bash
./docker.sh start
```

Access Jupyter Notebook at:

```bash
http://localhost:8888
```

Copy the **access token** displayed in the terminal to log in.

To open a shell inside the running container:

```bash
./docker.sh exec
```

To stop and clean up the container:

```bash
./docker.sh remove
```