import os

# set CUDA_MODULE_LOADING=LAZY to speed up the serverless function
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# set SAFETENSORS_FAST_GPU=1 to speed up the serverless function
os.environ["SAFETENSORS_FAST_GPU"] = "1"
import inspect
import imageio
import tempfile
import numpy as np
from omegaconf import OmegaConf
from safetensors import safe_open
from einops import rearrange

import torch
import transformers

transformers.logging.set_verbosity_error()
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.util import apply_lora, apply_motion_lora


def save_video(frames: torch.Tensor, seed=""):
    # save seed to the file name, for reproducibility
    output_video_path = tempfile.NamedTemporaryFile(prefix="{}_".format(seed), suffix=".mp4").name
    frames = (rearrange(frames, "b c t h w -> t b h w c").squeeze(1).cpu().numpy() * 255).astype(np.uint8)
    writer = imageio.get_writer(output_video_path, fps=8, codec="libx264", quality=9, pixelformat="yuv420p", macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return output_video_path


class AnimateDiff:
    def __init__(self, version="v2"):
        assert version in ["v1", "v2"], "version must be either v1 or v2"
        pretrained_model_path = os.path.join(os.path.dirname(__file__), "models/StableDiffusion/stable-diffusion-v1-5")
        motion_module = os.path.join(os.path.dirname(__file__), "models/Motion_Module/mm_sd_v15_{}-fp16.safetensors".format(version))

        self.inference_config = OmegaConf.load(os.path.join(os.path.dirname(__file__), "inference_{}.yaml".format(version)))
        self.guidance_scale = 7.5

        self.person_prompts = ["boy", "girl", "man", "woman", "person", "eye", "face"]

        # can be changed
        self.width  = 512
        self.height = 768

        # can not be changed
        self.video_length = 16
        self.use_fp16 = True
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        *_, func_args = inspect.getargvalues(inspect.currentframe())
        func_args = dict(func_args)

        tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", torch_dtype=self.dtype)
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", torch_dtype=self.dtype)
        vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", torch_dtype=self.dtype)
        unet         = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(self.inference_config.Model.unet_additional_kwargs),
            device=self.device,
            torch_dtype=self.dtype,
        )

        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            assert False

        self.pipeline = AnimationPipeline(
            vae          = vae,
            text_encoder = text_encoder,
            tokenizer    = tokenizer,
            unet         = unet,
            scheduler    = DDIMScheduler(
                **OmegaConf.to_container(self.inference_config.Model.noise_scheduler_kwargs), steps_offset=1, clip_sample=False
            ),
        )
        self.current_model = ""  # Person or Scene

        # 1.1 motion module
        motion_module_state_dict = {}
        with safe_open(motion_module, framework="pt", device=self.device) as f:
            for key in f.keys():
                motion_module_state_dict[key] = f.get_tensor(key)
        if "global_step" in motion_module_state_dict:
            func_args.update({"global_step": motion_module_state_dict["global_step"]})
        missing, unexpected = self.pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
        assert len(unexpected) == 0, f"Unexpected keys in motion module: {unexpected}"

    def _update_model(self, select):
        assert select in ["Person", "Scene"], "select must be either Person or Scene"
        if select == "Person":
            model_config = self.inference_config.Person
        elif select == "Scene":
            model_config = self.inference_config.Scene
        # skip if the model is already loaded
        if select == self.current_model:
            return -1

        # 1.2 T2I
        if model_config.base != "":
            if model_config.base.endswith(".ckpt"):
                state_dict = torch.load(model_config.base, map_location=self.device)
                self.pipeline.unet.load_state_dict(state_dict)

            elif model_config.base.endswith(".safetensors"):
                if model_config.lora is None:
                    is_lora = False
                else:
                    is_lora = len(model_config.lora) != 0
                if is_lora:
                    lora_state_dicts = []
                    for lora_name in model_config.lora:
                        lora_path, lora_scale = model_config.lora[lora_name]
                        # print(f"Loading {lora_name} from {lora_path} with scale {lora_scale}")
                        state_dict = {}
                        with safe_open(lora_path, framework="pt", device=self.device) as f:
                            for key in f.keys():
                                state_dict[key] = f.get_tensor(key)
                        lora_state_dicts.append((lora_name, state_dict, lora_scale))
                    # is_lora = all("lora" in k for k in state_dict.keys())
                else:
                    pass
                base_state_dict = {}
                with safe_open(model_config.base, framework="pt", device=self.device) as f:
                    for key in f.keys():
                        base_state_dict[key] = f.get_tensor(key)

                # vae
                converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, self.pipeline.vae.config)
                try:
                    self.pipeline.vae.load_state_dict(converted_vae_checkpoint)
                except:
                    converted_vae_checkpoint = {
                        key.replace(".query.", ".to_q.")
                        .replace(".key.", ".to_k.")
                        .replace(".value.", ".to_v.")
                        .replace(".proj_attn.", ".to_out.0."): value
                        for key, value in converted_vae_checkpoint.items()
                    }
                    self.pipeline.vae.load_state_dict(converted_vae_checkpoint)
                # unet
                converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, self.pipeline.unet.config)
                self.pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                # text_model
                self.pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)
                del base_state_dict

                # import pdb
                # pdb.set_trace()
                self.pipeline.to(self.device, self.dtype)

                if is_lora:
                    for lora_name, state_dict, lora_scale in lora_state_dicts:
                        self.pipeline = apply_lora(self.pipeline, state_dict, dtype=self.dtype, scale=lora_scale)
                        print(f"lora {lora_name} applied")
                    del lora_state_dicts

                motion_lora = model_config.motion_lora
                if motion_lora:
                    motion_lora_path, motion_lora_scale = model_config.motion_lora
                    if motion_lora_path.endswith(".ckpt"):
                        state_dict = torch.load(motion_lora_path, map_location=self.device)
                        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
                    elif motion_lora_path.endswith(".safetensors"):
                        state_dict = {}
                        with safe_open(motion_lora_path, framework="pt", device=self.device) as f:
                            for key in f.keys():
                                state_dict[key] = f.get_tensor(key)
                    else:
                        return -1
                    self.pipeline = apply_motion_lora(self.pipeline, state_dict, dtype=self.dtype, scale=motion_lora_scale)
                    del state_dict

        # update last_model to avoid loading the same model repeatedly
        self.current_model = select

    def inference(self, prompt, steps=28, width=None, height=None, seed=None):
        isPerson = False
        for keyword in self.person_prompts:
            if keyword in prompt:
                isPerson = True
                break
        if prompt.endswith("."):
            prompt = prompt[:-1] + ","
        else:
            prompt += ", "

        # update model
        if isPerson:
            model_config = self.inference_config.Person
            prompt += model_config.prompt
            self._update_model("Person")
        else:
            model_config = self.inference_config.Scene
            prompt += model_config.prompt
            self._update_model("Scene")

        # if specified, use the specified width and height
        width  = width if width is not None else self.width
        height = height if height is not None else self.height

        # if not isPerson, reverse the size
        if not isPerson:
            width, height = height, width

        # inference
        n_prompt = model_config.n_prompt
        torch.seed()
        seed = seed if seed is not None else torch.randint(0, 1000000000, (1,)).item()
        torch.manual_seed(seed)

        print(f"current seed: {torch.initial_seed()}")
        print(f"sampling {prompt} ...")
        print(f"negative prompt: {n_prompt}")
        with torch.no_grad():
            sample = self.pipeline(
                prompt              = prompt,
                negative_prompt     = n_prompt,
                num_inference_steps = steps,
                guidance_scale      = self.guidance_scale,
                width               = width,
                height              = height,
                video_length        = self.video_length,
            ).videos

            save_path = save_video(sample, seed=seed)
        return save_path


if __name__ == "__main__":
    # example seeds:
    # Person: 445608568
    # Scene : 195577361
    fixed_seed = 195577361  # torch.randint(0, 1000000000, (1,)).item()

    animate_diff = AnimateDiff()
    import json
    with open("test_input.json", "r") as f:
        test_input = json.load(f)["input"]

    # faster config
    save_path = animate_diff.inference(test_input["prompt"], test_input["steps"], test_input["width"], test_input["height"], seed=fixed_seed)
    print("Result of faster config is saved to: {}\n".format(save_path))

    # better config
    save_path = animate_diff.inference(test_input["prompt"], 28, 512, 768, seed=fixed_seed)
    print("Result of better config is saved to: {}\n".format(save_path))

    # standard config, same to the original repo
    save_path = animate_diff.inference(test_input["prompt"], 25, 512, 512, seed=fixed_seed)
    print("Result of standard config is saved to: {}\n".format(save_path))
