import os
from huggingface_hub import hf_hub_download
from gdown import download as gdown_download

# Motion modules are the fp16+safetensors versions converted by the author of this repo.
# They are faster to be loaded compared to the originals, which is important for serverless usage.
# Other models are the original models from huggingface.
with open(os.path.join(os.path.dirname(__file__), "download_list.txt"), "r") as r:
    download_list = r.read().splitlines()

for download in download_list:
    print(download)
    download_from, info, download_to = download.split(" ")
    if download_from == "huggingface":
        print(info)
        repo_id = info.split("/")[0] + "/" + info.split("/")[1]
        subfolder = info.split("/")[2] if len(info.split("/")) > 3 else None
        filename = info.split("/")[-1]
        save_path = os.path.join(os.path.dirname(__file__), "..", download_to)
        local_dir = os.path.dirname(download_to)
        local_dir = os.path.dirname(local_dir) if subfolder is not None else local_dir
        local_dir = os.path.join(os.path.dirname(__file__), "..", local_dir)
        print("Downloading from {}/{}/{} to {}".format(repo_id, subfolder, filename, local_dir))
        hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, local_dir=local_dir, local_dir_use_symlinks=False)
        if filename == os.path.basename(save_path):
            # no need to move
            pass
        else:
            os.rename(os.path.join(local_dir, filename), save_path)
    elif download_from == "gdrive":
        gdown_download(id=info, output=os.path.join(os.path.dirname(__file__), "..", download_to), quiet=False, use_cookies=False)
    else:
        raise Exception("Unknown download source: {}".format(download_from))