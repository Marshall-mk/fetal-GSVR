# Create a "tag" or name for the image
docker_tag=aicregistry:5000/${USER}:gsvr

# Delete existing Docker image if it exists
docker image rm -f ${docker_tag} || true

# Build the image from scratch
docker build --no-cache . -f Dockerfile \
    --tag ${docker_tag} --network=host \
    --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg USER=${USER}

docker push ${docker_tag}