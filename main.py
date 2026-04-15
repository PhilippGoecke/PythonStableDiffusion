from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, DiffusionPipeline
import torch

# Configuration
MODEL_ID = "stabilityai/sdxl-turbo"
# Alternative models:
# MODEL_ID = "stabilityai/stable-diffusion-3-medium"
# MODEL_ID = "black-forest-labs/FLUX.1-dev"
# MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# MODEL_ID = "runwayml/stable-diffusion-v1-5"

PROMPT = "A futuristic cyberpunk city at night with neon lights, detailed, 4k, cinematic lighting, rain reflections, flying cars, holographic billboards, cyberpunk aesthetic"
OUTPUT_FILE = "output.png"

# Map models to their pipeline classes
MODEL_PIPELINES = {
    "stabilityai/sdxl-turbo": StableDiffusionXLPipeline,
    "stabilityai/stable-diffusion-xl-base-1.0": StableDiffusionXLPipeline,
    "stabilityai/stable-diffusion-3-medium": DiffusionPipeline,
    "black-forest-labs/FLUX.1-dev": DiffusionPipeline,
    "runwayml/stable-diffusion-v1-5": StableDiffusionPipeline,
}

# Select pipeline class
pipeline_class = MODEL_PIPELINES.get(MODEL_ID, DiffusionPipeline)

# Determine device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
elif torch.version.hip is not None:
    device = "cuda"

# Load pipeline
pipe = pipeline_class.from_pretrained(MODEL_ID, torch_dtype=torch.float16, variant="fp16")
pipe.to(device)

# Generate and save
image = pipe(PROMPT).images[0]
image.save(OUTPUT_FILE)
