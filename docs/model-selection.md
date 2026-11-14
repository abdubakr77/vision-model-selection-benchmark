# Model selection

Status: Phase 1 candidates. Everything here is provisional until the models run.

| Family | Phase 1 candidate | Notes |
|---|---|---|
| YOLO / detection | Ultralytics YOLO (small variant) | AGPL-3.0, enterprise license exists. Flag in the decision framework. |
| Vision Transformer | Swin or ViT classifier (pretrained, fine-tuned) | fp16 on T4 |
| Open-source VLM | Gemma 3 4B-it or Qwen2-VL-2B | zero-shot, JSON output |
| Gemma-family | Gemma 3 4B-it (gated on Hugging Face) | accept license, HF token in Kaggle Secrets. Check newer Gemma releases first. |
| Cloud vision model | Gemini via Google AI Studio free tier | Phase 2 |
| Local / private | same open models run on own hardware | compare cost and privacy against cloud |

Reference compute: Kaggle T4 (about 16 GB VRAM), fp16, 4-bit quantization only where needed.
