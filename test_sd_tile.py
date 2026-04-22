import torch
import os
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler

def main():
    device = "cuda"
    # 统一使用与 download_weights.py 一致的相对路径
    sd_path = "pretrained_weights/sd15"
    controlnet_path = "pretrained_weights/controlnet_tile"
    input_path = "results/gan_output_010.png"
    output_dir = "results"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("input_samples", exist_ok=True)

    # 【防呆设计】如果没有测试图，自动生成一张 512x512 的测试图防止程序退出
    if not os.path.exists(input_path):
        print(f"⚠️ 未找到 {input_path}，已自动生成一张测试图以供跑通流程。")
        Image.new('RGB', (512, 512), color='gray').save(input_path)

    print("🔄 正在加载标准版 ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, 
        torch_dtype=torch.float16,
        use_safetensors=True  # 你下载的正是 safetensors，这里会完美匹配
    )

    print("🔄 正在加载 Stable Diffusion v1.5...")
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        sd_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)

    # 使用推荐的调度器
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # 加载图片
    init_image = Image.open(input_path).convert("RGB")

    print("🚀 开始推理...")
    image = pipe(
        prompt="high quality, detailed, masterpiece",
        negative_prompt="blur, low quality, distortion, artifacts",
        image=init_image,
        control_image=init_image,
        strength=0.5,           
        num_inference_steps=20,
        controlnet_conditioning_scale=1.0,
    ).images[0]

    image.save(os.path.join(output_dir, "clean_rebuild_result.png"))
    print("✅ 任务完成！图已存入 results 文件夹。")

if __name__ == "__main__":
    main()
