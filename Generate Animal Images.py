from diffusers import StableDiffusionPipeline
import torch
import re

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
dtype = torch.float16 if use_cuda else torch.float32

print("CUDA available:", use_cuda)
print("Using device:", device)

pipe = StableDiffusionPipeline.from_pretrained(
    "animal_text2img_project/trained_model",
    torch_dtype=dtype
).to(device)

prompts = ["A man Holding a cat"]

for i, prompt in enumerate(prompts, start=1):
    image = pipe(prompt, num_inference_steps=20).images[0]
    safe_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', prompt.strip())[:30]
    image.save(f"{i}_{safe_name}.png")
    print(f"Saved: {i}_{safe_name}.png")