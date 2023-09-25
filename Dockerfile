# Include base image
FROM docker.io/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Define working directory
WORKDIR /workspace/

# Set timezone
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies
RUN apt-get update && apt-get -y install libgl1 libglib2.0-0 vim
RUN apt-get autoremove -y && apt-get clean -y

# Add pretrained model
ADD animatediff ./animatediff
ADD models ./models

# Add necessary files
ADD inference_v1.yaml ./
ADD inference_v2.yaml ./
ADD inference_util.py ./
ADD server.py ./

# pip install
ADD requirements.txt ./
RUN pip install -r requirements.txt

# Run server
CMD [ "python", "-u", "./server.py" ]
# ADD ./test_input.json ./
