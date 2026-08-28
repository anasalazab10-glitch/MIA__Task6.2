# Image Caption Generator (Flickr8k) — ResNet50 + LSTM

Production-oriented image caption generator trained on the **Flickr8k** dataset (8,000 images, 5 captions per image).
It uses **transfer learning** (ResNet50 pretrained on ImageNet) for visual features and an **LSTM decoder** for caption generation.

## Dataset
- Flickr8k (Kaggle): https://www.kaggle.com/datasets/adityajn105/flickr8k?select=captions.txt
- Each image has **5 captions**
- Split strategy: **train/val/test split by image** (to avoid leakage)

## Preprocessing
### Captions
- Lowercasing + cleanup
- Tokens: `<start>`, `<end>`, `<pad>`, `<unk>`
- Vocabulary built from **train only**, `min_freq=2`
- Max length: `20` (95th percentile, includes `<start>/<end>`)

### Images
- ResNet50 ImageNet preprocessing
- Extract cached **2048-d** features per image (improves training speed)

## Architecture
Encoder (fixed):
- ResNet50 backbone → 2048-d image feature vector

Decoder:
- Linear projection → initialize LSTM hidden state
- Word embedding (256)
- LSTM (hidden size 256)
- Linear layer → vocab logits
- Decoding: greedy decoding

## Training
- Teacher forcing (next-token prediction)
- Loss: CrossEntropy with padding ignored (`ignore_index=<pad>`)
- Adam + LR scheduling + early stopping + best checkpoint

## Evaluation (Test Set)
- BLEU-1: 0.5647
- BLEU-2: 0.3731
- BLEU-3: 0.2453
- BLEU-4: 0.1685
- ROUGE-L (F1): 0.4350
- METEOR: 0.3678

## Qualitative example
**Input image:** `1007320043_627395c3d8.jpg`  
**Generated:** `two little girls are playing in a playground`  
**References:**
- a child playing on a rope net
- a little girl climbing on red roping
- a little girl in pink climbs a rope bridge at the park
- a small child grips onto the red ropes at the playground
- the small child climbs on a red ropes on a playground

## Model Artifacts (Hugging Face)

**Attention model (default / best):**
https://huggingface.co/Anas1010/flickr8k-image-caption-generator-attn

**Baseline model (older):**
https://huggingface.co/Anas1010/flickr8k-image-caption-generator

## Run locally (FastAPI)

```powershell
cd image-caption-generator
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .

$env:ICG_DEVICE="cpu"   # or "cuda" if available
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000

then open:

http://127.0.0.1:8000/docs
