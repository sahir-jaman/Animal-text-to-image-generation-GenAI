# test_model.py
import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path
import os

# Check GPU
if not torch.cuda.is_available():
    print("GPU not available!")
    exit(1)

print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")

# Load your fine-tuned model - Convert Path to string
model_path = "animal_text2img_project/trained_model"  # Use string, not Path object
print(f"✓ Loading model from {model_path}")

# Try to load with error handling
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).to("cuda")
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("\nTrying with safety checker disabled...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None  # Disable safety checker if causing issues
    ).to("cuda")
    print("✓ Model loaded with safety checker disabled!")

# Enable memory efficient attention
pipe.enable_attention_slicing()

# Optional: Enable memory efficient attention for better performance
if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✓ Enabled xformers memory efficient attention")
    except:
        print("⚠️ xformers not available, using standard attention")

# Generate some animal images
prompts = [
    "a cute brown dog running in a sunny park",
    "a fluffy white cat sitting on a windowsill",
    "a majestic horse standing in a green meadow",
    "a small bird perched on a tree branch",
    "two puppies playing in the grass"
]

print("\n🎨 Generating images with your fine-tuned model...")
output_dir = Path("generated_animals")
output_dir.mkdir(exist_ok=True)

for i, prompt in enumerate(prompts, 1):
    print(f"\nGenerating {i}/{len(prompts)}: {prompt[:50]}...")
    
    try:
        with torch.autocast("cuda"):
            image = pipe(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
        
        output_path = output_dir / f"animal_{i}.png"
        image.save(output_path)
        print(f"  ✓ Saved to {output_path}")
    except Exception as e:
        print(f"  ✗ Error generating image: {e}")

print(f"\n✅ All images saved to {output_dir}/")