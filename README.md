# AnimateDiff_Serverless_Runpod

## 1. Introduction
This is a serverless application that uses [AnimateDiff](https://animatediff.github.io/) to run a **Text-to-Video** task on [RunPod](https://www.runpod.io/).

See also [SDXL_Serverless_Runpod](https://github.com/sky24h/SDXL_Serverless_Runpod) for **Text-to-Imgae** task.

Serverless means that you are only charged for the time you use the application, and you don't need to pay for the idle time, which is very suitable for this kind of application that is not used frequently but needs to respond quickly.

Theoretically, this application can be called by any other application. Here we provide two examples:
1. A simple Python script
2. A Telegram bot

See [Usage](#Usage) below for more details.

### Example Result:
Input Prompt:
(random seed: 445608568)
```
1girl, focus on face, offshoulder, light smile, shiny skin, best quality, masterpiece, photorealistic
```

Result:
(Original | PanLeft, 28 steps, 768x512, around 60 seconds on RTX 3090, 0.015$😱 on RunPod)


https://github.com/sky24h/AnimateDiff_Serverless_Runpod/assets/26270672/b15fb186-b9a3-4077-b212-4b0c22e02dd1




Input Prompt:
(random seed: 195577361)
```
photo of coastline, rocks, storm weather, wind, waves, lightning, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3
```

Result:
(Original | ZoomOut, 28 steps, 768x512, around 60 seconds on RTX 3090, 0.015$😱 on RunPod)


https://github.com/sky24h/AnimateDiff_Serverless_Runpod/assets/26270672/563f020f-cd65-433f-8ac9-ee4af4c9d1f9




#### Time Measurement Explanation:
The time is measured from the moment the input prompt is sent to the moment the result image is received, including the time for all the following steps:
- Receive the request from the client
- Serverless container startup
- Model loading
- Inference
- Sending the result image back to the client.

## 2. Dependencies
- Python >= 3.9
- Docker
- Local GPU is necessary for testing but not necessary for deployment. (Recommended: RTX 3090)

If you don't have a GPU, you can modify and test the code on [Google Colab](https://colab.research.google.com/) and then build and deploy the application on RunPod.

Example Notebook: [link](https://colab.research.google.com/drive/1Gd6uuiItbIFjVPFNyJQhEEEL9khdAyY7?usp=sharing)

<a id="Usage"></a>
## 3. Usage
#### 1. Test on Local Machine
```
# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download.py

# Edit (or not) config to customize your inference, e.g., change base model, lora model, motion lora model, etc.
rename inference_v2(example).yaml to inference_v2.yaml

# Run inference test
python inference_util.py

# Run server.py local test
python server.py
```

During downloading, if you encounter errors like "gdown.exceptions.FileURLRetrievalError: Cannot retrieve the public link of the file.", 
reinstalling the gdown package using "pip install --upgrade --no-cache-dir gdown" and rerunning the download.py may help.


#### 2. Deploy on RunPod
1. First, make sure you have installed Docker and have accounts on both DockerHub and RunPod.

2. Then, decide a name for your Docker image, e.g., "your_username/anidiff:v1" and set your image name in "./scripts/build.sh".

3. Run the following commands to build and push your Docker image to DockerHub.

bash scripts/build.sh


4. Finally, deploy your application on RunPod to create [Template](https://docs.runpod.io/docs/template-creation) and [Endpoint](https://docs.runpod.io/docs/autoscaling).

Sorry for not providing detailed instructions here as the author is quite busy recently. You can find many detailed instructions on Google about how to deploy a Docker image on RunPod.

Feel free to contact me if you encounter any problems after searching on Google.

#### 3. Call the Application
##### Call from a Python script
```
# Make sure to set API key and endpoint ID before running the script.
python test_client.py
```

##### Showcase: Call from a Telegram bot
![Example Result](./assets/telegram_bot_example.jpg)

## 4. TODO
- [x] Support for specific base model for different objectives. (Person and Scene)
- [x] Support for LoRA models. (Edit yaml file and place your model in "./models/DreamBooth_LoRA")
- [x] Support for Motion LoRA models. (Also editable in yaml file, see [here](https://github.com/guoyww/AnimateDiff#features) for details and downloads.)
- [ ] More detailed instructions
- [ ] One-click deploy (If anyone is interested...)

## 4. Acknowledgement
Thanks to [AnimateDiff](https://animatediff.github.io/) and [RunPod](https://www.runpod.io/).
