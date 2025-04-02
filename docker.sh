#!/bin/bash

# Shell script for managing the ML4D Docker container

IMAGE_NAME="ml4d-env"
CONTAINER_NAME="ml4d-container"
WORKSPACE_DIR="$(pwd)"
DEFAULT_PORT=9999

# Function to show help
show_help() {
    echo "Usage: ./docker.sh [COMMAND]"
    echo "Commands:"
    echo "  build      Build the Docker image"
    echo "  start      Start the Docker container"
    echo "  exec       Execute a shell inside the running container"
    echo "  jupyter     Run Jupyter Notebook server in the container"
    echo "  remove     Stop and remove the container"
    echo "  help       Show this help message"
}

# Function to build the Docker image
build_image() {
    echo "Building Docker image: $IMAGE_NAME"
    docker build -t $IMAGE_NAME .
}

# Function to start the Docker container
start_container() {
    echo "Starting Docker container: $CONTAINER_NAME"
    docker run --gpus all --rm -dit \
        --name $CONTAINER_NAME \
        -v $WORKSPACE_DIR:/workspace \
        $IMAGE_NAME
    echo "Container started."
}

# Function to execute a shell inside the container
exec_shell() {
    echo "Opening a shell in the container: $CONTAINER_NAME"
    docker exec -it $CONTAINER_NAME /bin/bash
}

# Function to run jupyter
run_jupyter() {
    echo "Starting Jupyter Notebook on http://localhost:$DEFAULT_PORT"
    docker run --gpus all --rm -it \
        --name $CONTAINER_NAME \
        -v $WORKSPACE_DIR:/workspace \
        -p $DEFAULT_PORT:$DEFAULT_PORT \
        $IMAGE_NAME \
        bash -c "cd /workspace && jupyter notebook --ip=0.0.0.0 --port=$DEFAULT_PORT --allow-root --NotebookApp.token='' --NotebookApp.password=''"
}

# Function to stop and remove the container
remove_container() {
    echo "Stopping and removing container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME
}

# Main script logic
case "$1" in
    build)
        build_image
        ;;
    start)
        start_container
        ;;
    exec)
        exec_shell
        ;;
    jupyter)
        run_jupyter
        ;;
    remove)
        remove_container
        ;;
    help)
        show_help
        ;;
    *)
        echo "Error: Invalid command"
        show_help
        exit 1
        ;;
esac
