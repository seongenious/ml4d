#!/bin/bash
# Shell script for managing the ML4D Docker container
IMAGE_NAME="ml4d-env"
CONTAINER_NAME="ml4d-container"
WORKSPACE_DIR="/home/$USER"
DEFAULT_PORT=9999

# Function to show help
show_help() {
    echo "Usage: ./docker.sh [COMMAND]"
    echo "Commands:"
    echo "  build      Build the Docker image"
    echo "  start      Start the Docker container"
    echo "  jupyter    Start jupyterLab inside the container"
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
    docker run -dit \
        --gpus all \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $WORKSPACE_DIR:/mnt \
        -p $DEFAULT_PORT:$DEFAULT_PORT \
        --name $CONTAINER_NAME \
        $IMAGE_NAME
    echo "Container started."
}

start_jupyter() {
    echo "Launching JupyterLab on http://localhost:$DEFAULT_PORT"
    docker exec -d $CONTAINER_NAME jupyter lab \
        --ip=0.0.0.0 --port=$DEFAULT_PORT --no-browser --allow-root \
        --NotebookApp.token='' --NotebookApp.password=''
    echo "JupyterLab launched."
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
    jupyter)
        start_jupyter
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
