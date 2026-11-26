import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='/workspace/outputs/hf'))
