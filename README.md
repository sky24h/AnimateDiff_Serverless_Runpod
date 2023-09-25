# AnimateDiff_Serverless_Runpod

## 1. Introduction
This is a serverless application that uses [AnimateDiff](https://animatediff.github.io/) to run a Text-to-Image task on [RunPod](https://www.runpod.io/).

Serverless means that you are only charged for the time you use the application, and you don't need to pay for the idle time, which is very suitable for this kind of application that is not used frequently but needs to respond quickly.

Theoretically, this application can be called by any other application. Here we provide two examples:
1. A simple Python script
2. A Telegram bot

See [Usage](#Usage) below for more details.

### Example Result:
Input Prompt:
```
1girl, offshoulder, light smile, shiny skin best quality, masterpiece, photorealistic
```

Result:
(28 steps, 768x512, around 60 seconds on RTX 3090, 0.015$😱 on RunPod)
https://github.com/sky24h/AnimateDiff_Serverless_Runpod/assets/26270672/3f71dbff-7b33-44ce-881d-e57e86ec3416



Input Prompt:
```
photo of coastline, rocks, storm weather, wind, waves, lightning, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3
```

Result:
(28 steps, 768x512, around 60 seconds on RTX 3090, 0.015$😱 on RunPod)
https://github.com/sky24h/AnimateDiff_Serverless_Runpod/assets/26270672/e8bf52ca-1d16-4e65-8e35-5d40c4a3b3ae





#### Time Measurement Explanation:
The time is measured from the moment the input prompt is sent to the moment the result image is received, including the time for all the following steps:
- Receive the request from the client
- Serverless container startup
- Model loading
- Inference
- Sending the result image back to the client.

## 2. Dependencies
- Python >= 3.9.7
- Docker
- Local GPU is necessary for testing but not necessary for deployment. (Recommended: RTX 3090)

If you don't have a GPU, you can modify and test the code on [Google Colab](https://colab.research.google.com/) and then build and deploy the application on RunPod.

<a id="Usage"></a>
## 3. Usage
#### 1. Test on Local Machine
```
# Install dependencies
pip install -r requirements.txt

# Download models
python script/download.py

# Edit (or not) config to customize your inference
rename inference_v2(example).yaml to inference_v2.yaml

# Run inference test
python inference_util.py

# Run server.py local test
python server.py
```

#### 2. Deploy on RunPod
1. First, make sure you have installed Docker and have accounts on both DockerHub and RunPod.

2. Then, decide a name for your Docker image, e.g., "your_username/anidiff:v1" and set your image name in "./scripts/build.sh".

3. Run the following commands to build and push your Docker image to DockerHub.

bash scripts/build.sh


4. Finally, deploy your application on RunPod to create [Template](https://docs.runpod.io/docs/template-creation) and [Endpoint](https://docs.runpod.io/docs/autoscaling).

Sorry for not providing detailed instructions here as the author is quite busy recently. You can find many detailed instructions on Google about how to deploy a Docker image on RunPod.

Feel free to contact me if you encounter any problems after searching on Google.

#### 3. Call the Application
##### Call the application from a Python script
```
python test_client.py
```

##### Showcase: Call the application from a Telegram bot
![Example Result](./assets/telegram_bot_example.jpg)

## 4. TODO
- [x] Support specific base model for different objectives. (Person and Scene)
- [x] Support LoRa models. (Edit yaml file and place your model in "./models/DreamBooth_LoRA")
- [ ] More detailed instructions
- [ ] One-click deploy (If anyone is interested...)

## 4. Acknowledgement
Thanks to [AnimateDiff](https://animatediff.github.io/) and [RunPod](https://www.runpod.io/).
