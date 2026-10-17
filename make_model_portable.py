from diffusers import StableDiffusionPipeline
import torch

model_path = "animal_text2img_project/trained_model"

pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float32
)

pipe.save_pretrained(model_path)

print("Full portable model saved successfully.")