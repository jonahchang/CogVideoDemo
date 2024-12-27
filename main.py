import torch
from transformers import T5EncoderModel
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXPipeline
from diffusers.utils import export_to_video
from torchao.quantization import quantize_, int8_weight_only

def generate_video(prompt: str, output_file: str = "output.mp4"):
    # Quantize components to reduce VRAM usage
    quantization = int8_weight_only

    # Load and quantize the text encoder
    text_encoder = T5EncoderModel.from_pretrained(
        "THUDM/CogVideoX-5b", 
        subfolder="text_encoder", 
        torch_dtype=torch.bfloat16
    )
    quantize_(text_encoder, quantization())

    # Load and quantize the transformer
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-5b", 
        subfolder="transformer", 
        torch_dtype=torch.bfloat16
    )
    quantize_(transformer, quantization())

    # Load and quantize the VAE
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b", 
        subfolder="vae", 
        torch_dtype=torch.bfloat16
    )
    quantize_(vae, quantization())

    # Initialize the pipeline with quantized components
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5b",
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )

    # Enable VRAM optimization
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    # Generate the video
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=100,  # Increase steps for better results
        num_frames=49,           # Keep frame count low for efficiency
        guidance_scale=6,
        generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42),
    ).frames[0]

    # Export the video
    export_to_video(video, output_file, fps=8)
    print(f"Video saved to {output_file}")

if __name__ == "__main__":
    # Example prompt
    video_prompt = (
        "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
        "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. "
        "Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. "
        "Sunlight filters through the tall bamboo, casting a gentle glow on the scene. "
        "The panda's face is expressive, showing concentration and joy as it plays. "
        "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere."
    )
    generate_video(prompt=video_prompt)
