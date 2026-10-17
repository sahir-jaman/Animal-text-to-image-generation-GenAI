# Animal Text-to-Image Generation (Fine-Tuned Stable Diffusion)

Type a sentence, get a picture of an animal. This project adapts a pre-trained **Stable Diffusion** model to the animal domain so that prompts like *"a brown dog running"* or *"a white cat sitting"* produce relevant, focused images.

| Prompt | Generated image |
|---|---|
| a brown dog running | <img src="samples/brown_dog_running.png" width="260"> |
| a white cat sitting | <img src="samples/white_cat_sitting.png" width="260"> |
| a red dog swimming | <img src="samples/red_dog_swimming.png" width="260"> |

## The problem

General-purpose text-to-image models are trained on everything, so for a narrow topic such as wildlife they can drift: wrong details, or unrelated objects (often people) appearing in the picture. Image-only datasets don't help here, because the model needs **image + caption pairs** to learn how words map to visuals.

## What I did

1. **Chose the right data.** An image-only dataset turned out to be unsuitable for text-conditioned generation, so I switched to **Flickr8k** (about 8,000 images, each with several captions).
2. **Filtered it to animals.** Keyword matching kept animal captions and removed human-centric ones, cutting roughly 40,000 captions down to about 9,000.
3. **Explored the data.** Caption counts, animal keyword frequency, caption length and sample images, to check that the filter worked.
4. **Split it** 70% train / 15% validation / 15% test.
5. **Fine-tuned Stable Diffusion** with Hugging Face Diffusers (512 x 512 images, learning rate 1e-5, batch size 1 with gradient accumulation).
6. **Compared** the fine-tuned model against the original pre-trained model on the same kind of prompts.

![Methodology](images/methodology.png)

## Results

- The fine-tuned model produced images that matched the prompts more closely, with better structure, less noise and fewer irrelevant elements than the baseline.
- Evaluation is **qualitative** (visual comparison of baseline and fine-tuned outputs). I did not compute FID or other numeric image metrics.
- The data is dominated by dogs, so the model does best on common animals and struggles more with rarer animals and complex prompts.

| Caption counts before and after filtering | Most common animal words |
|---|---|
| <img src="images/caption_count_before_after.png" width="380"> | <img src="images/animal_keyword_distribution.png" width="380"> |

![Sample filtered images](images/sample_filtered_images.png)

## Repository contents

| File | Purpose |
|---|---|
| `train_model.py` | Fine-tunes the diffusion model on the filtered image-caption pairs |
| `Generate Animal Images.py` | Generates images from text prompts |
| `test_model.py` | Tests the model on sample prompts |
| `make_model_portable.py` | Packages the trained model for portable inference |
| `captions.txt` | Captions used for training |
| `requirements.txt` | Python dependencies |
| `samples/` | Example generated images |

The training images and model weights are not included because of their size.

## Quick start

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python train_model.py           # fine-tune (a GPU is strongly recommended)
python "Generate Animal Images.py"
```

Python 3.12 is recommended, because some pinned packages have no builds for the newest Python versions.

## Limitations and next steps

- Small, dog-heavy dataset, so rarer animals are weaker.
- Keyword filtering is simple and can miss or wrongly keep some captions.
- Add quantitative evaluation (FID, CLIP score) and a larger, more balanced dataset.

## Author

Sahir Zaman: [LinkedIn](https://www.linkedin.com/in/sahir-zaman-338632190) | [GitHub](https://github.com/sahir-jaman)
