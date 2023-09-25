docker_image_name={"your_docker_image_name"}
docker build . --tag=$docker_image_name
docker push $docker_image_name
