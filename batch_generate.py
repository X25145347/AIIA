import os
from diffusers import StableDiffusionPipeline
import torch

# Load model (SD 1.5 recommended for low RAM)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)
pipe.to("cpu")

# Create output folder
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Load prompts
with open("prompts.txt", "r") as f:
    prompts = [line.strip() for line in f.readlines() if line.strip()]
    
# Generate images
for i, prompt in enumerate(prompts):
    print(f"Generating image {i+1}/{len(prompts)}: {prompt}")
    image = pipe(prompt).images[0]
    filename = os.path.join(output_dir, f"image_{i:04d}.png")
    image.save(filename)

print("Batch generation complete.")