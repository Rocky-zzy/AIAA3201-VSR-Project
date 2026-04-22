import os
import warnings
from huggingface_hub import snapshot_download

warnings.filterwarnings("ignore")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def download_optimized():
    # 路径已经通过软链接指向 /data1 了
    base_path = "./pretrained_weights"
    
    print("🚀 正在下载精简版 SD 1.5 (只保留 fp16)...")
    snapshot_download(
        repo_id="runwayml/stable-diffusion-v1-5",
        local_dir=os.path.join(base_path, "sd15"),
        # 核心优化：只允许下载 fp16 权重和必要的配置文件
        allow_patterns=["*.fp16.safetensors", "*.json", "*.txt", "unet/*", "vae/*", "tokenizer/*", "scheduler/*", "text_encoder/*"],
        ignore_patterns=["*.msgpack", "*.ckpt", "*.onnx", "pt_model.bin"],
        resume_download=True
    )

    print("\n🚀 正在下载 ControlNet Tile...")
    snapshot_download(
        repo_id="lllyasviel/control_v11f1e_sd15_tile",
        local_dir=os.path.join(base_path, "controlnet_tile"),
        allow_patterns=["*.safetensors", "*.json"],
        resume_download=True
    )
    print("\n✅权重已保存至 /data1！")

if __name__ == "__main__":
    download_optimized()