import torch
import sys

# Check if running on GPU
if not torch.cuda.is_available():
    print("ERROR: This code requires a GPU to run!")
    print("No CUDA-capable GPU detected. Please run this on a machine with GPU support.")
    sys.exit(1)

print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
print(f"✓ CUDA Version: {torch.version.cuda}")
print(f"✓ Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("-" * 50)

# Import libraries with progress indicators
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from pathlib import Path
import shutil
import json
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("✓ Libraries imported successfully")

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"✓ Random seeds set to {SEED}")

# Plot style
sns.set_theme(style="whitegrid", context="notebook")

# Paths
IMAGES_DIR = Path("Images")
CAPTIONS_FILE = Path("captions.txt")   # change if needed

# Output folders
BASE_DIR = Path("animal_text2img_project")
FILTERED_DIR = BASE_DIR / "filtered_images"
DATASET_DIR = BASE_DIR / "sd_dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "trained_model"
APP_DIR = BASE_DIR / "streamlit_app"

print("\n📁 Creating directories...")
for folder in [BASE_DIR, FILTERED_DIR, DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, RESULTS_DIR, MODEL_DIR, APP_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
print("✓ Setup complete")
print(f"✓ Images folder exists: {IMAGES_DIR.exists()}")
print(f"✓ Captions file exists: {CAPTIONS_FILE.exists()}")

def load_flickr8k_captions(captions_path):
    """
    Supports common formats:
    1) image,caption
    2) image#0 caption
    """
    print("\n📖 Loading captions...")
    rows = []

    with open(captions_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Remove header if present
    if lines[0].lower().startswith("image") or lines[0].lower().startswith("filename"):
        lines = lines[1:]

    for line in tqdm(lines, desc="Processing captions"):
        if "," in line:
            parts = line.split(",", 1)
            image_name = parts[0].strip()
            caption = parts[1].strip()
        else:
            # old Flickr8k token format: 1000268201_693b08cb0e.jpg#0 some caption
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            image_name = parts[0].split("#")[0].strip()
            caption = parts[1].strip()

        rows.append({"image_name": image_name, "caption": caption})

    df = pd.DataFrame(rows)
    print(f"✓ Loaded {len(df)} caption rows")
    print(f"✓ Unique images: {df['image_name'].nunique()}")
    return df

captions_df = load_flickr8k_captions(CAPTIONS_FILE)

animal_keywords = [
    "dog", "dogs", "puppy", "puppies",
    "cat", "cats", "kitten", "kittens",
    "horse", "horses",
    "bird", "birds",
    "cow", "cows",
    "sheep",
    "goat", "goats",
    "animal", "animals"
]

person_keywords = [
    "man", "woman", "boy", "girl", "child", "children",
    "person", "people", "human", "lady", "gentleman",
    "baby", "babies", "rider", "player", "skier", "surfer"
]

def contains_any(text, keywords):
    text = text.lower()
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in keywords)

print("\n🔍 Filtering animal images...")
captions_df["has_animal"] = captions_df["caption"].apply(lambda x: contains_any(x, animal_keywords))
captions_df["has_person"] = captions_df["caption"].apply(lambda x: contains_any(x, person_keywords))

# strict filter: animal-related and not person-heavy
animal_df = captions_df[(captions_df["has_animal"]) & (~captions_df["has_person"])].copy()

print(f"✓ Filtered caption rows: {len(animal_df)}")
print(f"✓ Filtered unique images: {animal_df['image_name'].nunique()}")

# Plotting with progress
print("\n📊 Generating visualizations...")
summary_df = pd.DataFrame({
    "Set": ["All captions", "Animal-filtered captions"],
    "Count": [len(captions_df), len(animal_df)]
})

plt.figure(figsize=(8, 5))
sns.barplot(data=summary_df, x="Set", y="Count")
plt.title("Caption Count Before and After Animal Filtering")
plt.ylabel("Number of captions")
plt.xlabel("")
plt.tight_layout()
plt.show()
print("✓ Caption count plot generated")

unique_summary_df = pd.DataFrame({
    "Set": ["All images", "Animal-filtered images"],
    "Count": [captions_df["image_name"].nunique(), animal_df["image_name"].nunique()]
})

plt.figure(figsize=(8, 5))
sns.barplot(data=unique_summary_df, x="Set", y="Count")
plt.title("Unique Image Count Before and After Filtering")
plt.ylabel("Number of unique images")
plt.xlabel("")
plt.tight_layout()
plt.show()
print("✓ Unique image count plot generated")

def extract_animal_terms(text):
    words = re.findall(r"\b[a-z]+\b", text.lower())
    return [w for w in words if w in animal_keywords]

all_terms = []
for caption in tqdm(animal_df["caption"], desc="Extracting animal terms"):
    all_terms.extend(extract_animal_terms(caption))

term_counts = pd.DataFrame(Counter(all_terms).most_common(15), columns=["term", "count"])

plt.figure(figsize=(10, 5))
sns.barplot(data=term_counts, x="term", y="count")
plt.title("Most Common Animal Terms in Filtered Captions")
plt.xlabel("Animal term")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("✓ Animal terms distribution plot generated")

animal_df["caption_length"] = animal_df["caption"].apply(lambda x: len(x.split()))

plt.figure(figsize=(10, 5))
sns.histplot(animal_df["caption_length"], bins=25, kde=True)
plt.title("Caption Length Distribution in Filtered Dataset")
plt.xlabel("Number of words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
print("✓ Caption length distribution plot generated")

print("\n🔍 Validating image files...")
valid_rows = []

for _, row in tqdm(animal_df.iterrows(), total=len(animal_df), desc="Checking images"):
    img_path = IMAGES_DIR / row["image_name"]
    if img_path.exists():
        valid_rows.append(row)

animal_df = pd.DataFrame(valid_rows).reset_index(drop=True)

print(f"✓ Valid filtered caption rows: {len(animal_df)}")
print(f"✓ Valid filtered unique images: {animal_df['image_name'].nunique()}")

# picking the first caption per image
image_level_df = (
    animal_df
    .sort_values("image_name")
    .groupby("image_name", as_index=False)
    .first()[["image_name", "caption"]]
    .copy()
)

image_level_df["image_path"] = image_level_df["image_name"].apply(lambda x: str(IMAGES_DIR / x))

print("\n📸 Displaying sample images...")
sample_rows = image_level_df.sample(9, random_state=SEED).reset_index(drop=True)

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

for ax, (_, row) in zip(axes, sample_rows.iterrows()):
    img = Image.open(row["image_path"]).convert("RGB")
    ax.imshow(img)
    ax.set_title(row["caption"][:60], fontsize=9)
    ax.axis("off")

plt.suptitle("Sample Animal-Filtered Flickr8k Images", fontsize=14)
plt.tight_layout()
plt.show()
print("✓ Sample images displayed")

print("\n📊 Splitting dataset...")
train_df, temp_df = train_test_split(
    image_level_df,
    test_size=0.30,
    random_state=SEED,
    shuffle=True
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    shuffle=True
)

print(f"✓ Train: {len(train_df)} images")
print(f"✓ Validation: {len(val_df)} images")
print(f"✓ Test: {len(test_df)} images")

split_summary = pd.DataFrame({
    "Split": ["Train", "Validation", "Test"],
    "Count": [len(train_df), len(val_df), len(test_df)]
})

plt.figure(figsize=(8, 5))
sns.barplot(data=split_summary, x="Split", y="Count")
plt.title("Train / Validation / Test Split Sizes")
plt.ylabel("Number of images")
plt.tight_layout()
plt.show()
print("✓ Split visualization generated")

train_df.to_csv(BASE_DIR / "train_split.csv", index=False)
val_df.to_csv(BASE_DIR / "val_split.csv", index=False)
test_df.to_csv(BASE_DIR / "test_split.csv", index=False)

print("✓ Split CSV files saved")

def prepare_split_folder(df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_dir / "metadata.jsonl"

    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Preparing {out_dir.name}"):
            src = Path(row["image_path"])
            dst = out_dir / row["image_name"]

            if not dst.exists():
                shutil.copy(src, dst)

            record = {
                "file_name": row["image_name"],
                "text": row["caption"]
            }
            meta_file.write(json.dumps(record) + "\n")

prepare_split_folder(train_df, TRAIN_DIR)
prepare_split_folder(val_df, VAL_DIR)
prepare_split_folder(test_df, TEST_DIR)

print("✓ Diffusers-style dataset folders created")

from diffusers import AutoPipelineForText2Image

print("\n🎨 Loading Stable Diffusion pipeline...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"✓ Using device: {device}")
if device == "cuda":
    print(f"✓ GPU Memory before loading model: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

pipe = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=dtype
).to(device)

print("✓ Baseline pipeline loaded successfully")
if device == "cuda":
    print(f"✓ GPU Memory after loading model: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

print("\n🖼️ Generating baseline images...")
baseline_prompts = [
    "a brown dog running on grass",
    "a white cat sitting on a chair",
    "a horse standing in a field"
]
baseline_paths = []

for i, prompt in enumerate(tqdm(baseline_prompts, desc="Generating images"), start=1):
    image = pipe(prompt, num_inference_steps=25).images[0]
    out_path = RESULTS_DIR / f"baseline_{i}.png"
    image.save(out_path)
    baseline_paths.append((prompt, out_path))

print("✓ Baseline images saved")

fig, axes = plt.subplots(1, len(baseline_paths), figsize=(15, 5))

for ax, (prompt, path) in zip(axes, baseline_paths):
    img = Image.open(path).convert("RGB")
    ax.imshow(img)
    ax.set_title(prompt, fontsize=9)
    ax.axis("off")

plt.suptitle("Baseline Stable Diffusion Outputs (Before Fine-Tuning)", fontsize=14)
plt.tight_layout()
plt.show()
print("✓ Baseline images displayed")

print("\n🔧 Setting up training environment...")
import subprocess
import sys

def run_command(command, description):
    print(f"\n{description}...")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ {description} completed successfully")
    else:
        print(f"✗ Error in {description}: {result.stderr}")
        return False
    return True

# Clone diffusers repo if not exists
if not Path("diffusers").exists():
    run_command("git clone https://github.com/huggingface/diffusers.git", "Cloning diffusers repository")
    run_command("cd diffusers && pip install -q .", "Installing diffusers")
    run_command("cd diffusers && pip install -q -r examples/text_to_image/requirements.txt", "Installing requirements")
else:
    print("✓ diffusers repository already exists")

from accelerate.utils import write_basic_config
write_basic_config()
print("✓ Accelerate config written")

print("\n🚀 Starting fine-tuning training...")
print("⚠️ This will take time depending on your GPU and dataset size")
print("=" * 50)

# Training command
train_cmd = """
accelerate launch diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --train_data_dir="animal_text2img_project/sd_dataset/train" \
  --image_column="image" \
  --caption_column="text" \
  --resolution=512 \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=5 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="animal_text2img_project/trained_model" \
  --validation_prompts="a brown dog running in a park" \
  --validation_epochs=1 \
  --checkpointing_steps=500 \
  --report_to="none"
"""

# Run training with progress display
process = subprocess.Popen(train_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

for line in process.stdout:
    print(line, end='')
    if "epoch" in line.lower() or "loss" in line.lower() or "step" in line.lower():
        sys.stdout.flush()

process.wait()

if process.returncode == 0:
    print("\n✅ Training completed successfully!")
    print(f"✓ Model saved to: {MODEL_DIR}")
else:
    print("\n❌ Training failed. Please check the error messages above.")

print("\n🎉 All tasks completed successfully!")