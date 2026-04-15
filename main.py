from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, DiffusionPipeline
import torch

# Configuration
MODEL_ID = "stabilityai/sdxl-turbo"
# Alternative models:
# MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# MODEL_ID = "runwayml/stable-diffusion-v1-5"
PROMPT = "A futuristic cyberpunk city at night with neon lights, detailed, 4k, cinematic lighting, rain reflections, flying cars, holographic billboards, cyberpunk aesthetic"
OUTPUT_FILE = "output.png"

MODEL_PIPELINES = {
    "stabilityai/sdxl-turbo": StableDiffusionXLPipeline,
    "stabilityai/stable-diffusion-xl-base-1.0": StableDiffusionXLPipeline,
    "runwayml/stable-diffusion-v1-5": StableDiffusionPipeline,
}

def get_device():
    """Determine best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def generate_image(model_id, prompt, output_file):
    """Generate and save image."""
    pipeline_class = MODEL_PIPELINES.get(model_id, DiffusionPipeline)
    device = get_device()

    pipe = pipeline_class.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to(device)

    image = pipe(prompt, height=256, width=256).images[0]
    image.save(output_file)
    print(f"Image saved to {output_file}")

if __name__ == "__main__":
    generate_image(MODEL_ID, PROMPT, OUTPUT_FILE)
