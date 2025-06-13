#!/bin/bash
# Shell script for managing the ML4D Docker container
IMAGE_NAME="ml4d-env"
CONTAINER_NAME="ml4d-container"
WORKSPACE_DIR="/home/$USER"

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

# Function to build the Docker image
build_image() {
    echo "Building Docker image: $IMAGE_NAME"
    docker build -t $IMAGE_NAME .
}

# Function to start the Docker container
start_container() {
    echo "Starting Docker container: $CONTAINER_NAME"
    docker run -dit \
        --gpus all \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $WORKSPACE_DIR:/mnt \
        --name $CONTAINER_NAME \
        $IMAGE_NAME
    echo "Container started."
}

# Function to execute a shell inside the container
into_shell() {
    echo "Opening a shell in the container: $CONTAINER_NAME"
    docker exec -it $CONTAINER_NAME /bin/bash
}

# Function to stop and remove the container
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
