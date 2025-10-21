#!/bin/bash
# Shell script for managing the ML4D Docker container
IMAGE_NAME="ml4d-env"
CONTAINER_NAME="ml4d-container"
WORKSPACE_DIR="/workspace"
DEFAULT_PORT=5000

# Function to show help
show_help() {
    echo "Usage: ./docker.sh [COMMAND]"
    echo "Commands:"
    echo "  build      Build the Docker image"
    echo "  start      Start the Docker container"
    echo "  exec       Execute a shell inside the running container"
    echo "  remove     Stop and remove the container"
    echo "  help       Show this help message"
}

build_image() {
    echo "Building Docker image: $IMAGE_NAME"
    docker build -t $IMAGE_NAME .
}

start_container() {
    echo "Starting Docker container: $CONTAINER_NAME"
    docker run -dit --gpus all \
        --shm-size=16g \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v ~/data/sets/nuscenes:/workspace/data/nuscenes \
        -v ~/git/ml4d/src:$WORKSPACE_DIR/src \
        -v ~/git/ml4d/data:$WORKSPACE_DIR/data \
        -v ~/git/ml4d/checkpoints:$WORKSPACE_DIR/checkpoints \
        -v ~/git/ml4d/logs:$WORKSPACE_DIR/logs \
        -v ~/git/ml4d/mlruns:$WORKSPACE_DIR/mlruns \
        -v ~/git/ml4d/scripts:$WORKSPACE_DIR/scripts \
        -p $DEFAULT_PORT:$DEFAULT_PORT \
        --name $CONTAINER_NAME \
        $IMAGE_NAME
    echo "Container started."
}

into_shell() {
    echo "Opening a shell in the container: $CONTAINER_NAME"
    docker exec -it $CONTAINER_NAME /bin/bash
}

remove_container() {
    echo "Stopping and removing container: $CONTAINER_NAME"
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME
    else
        echo "Container $CONTAINER_NAME does not exist."
    fi
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
        into_shell
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
