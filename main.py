from diffusers import StableDiffusionXLPipeline
import torch

model_id = "stabilityai/sdxl-turbo"

pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")

device = "cpu" # Set device to CPU (will be overridden if GPU is available)
if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPUs
elif torch.backends.mps.is_available():
    device = "mps"  # Apple Metal Performance Shaders
elif torch.version.hip is not None:
    device = "cuda"  # AMD GPUs use HIP, which maps to CUDA device

prompt = "A futuristic cyberpunk city at night with neon lights, detailed, 4k, cinematic lighting, rain reflections, flying cars, holographic billboards, cyberpunk aesthetic"

image = pipe(prompt).images[0]
image.save("output.png")
#image.show()
