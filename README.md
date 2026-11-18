# Holographic Phase-Key Honeypot Detection (ViT)

Detects Fourier-domain phase watermarks in digital holograms using a fine-tuned Vision Transformer.

### 1. Install Dependencies

```bash
git clone [https://github.com/username/holographic-vit-honeypot.git](https://github.com/username/holographic-vit-honeypot.git)
cd holographic-vit-honeypot
pip install torch numpy timm datasets scikit-learn matplotlib

```

### 2. Run the Pipeline

Set your Hugging Face token (optional, avoids rate limits) and run the training script or notebook:

```bash
export HF_TOKEN="your_hf_token"
python -m src.train --num_samples 300 --alpha 0.60 --epochs 15

```

## Pipeline Overview

1. **Data**: Streams 300 clean $224 \times 224$ complex holograms from `gokhankocmarli/inline-digital-holography-v3`.
2. **Watermark**: Injects a token-seeded (`HONEY_TOKEN_01`) phase mask into the high-frequency FFT band ($40 \le r \le 90$) with strength $\alpha = 0.60$.
3. **Features**: Extracts a 3-channel tensor (`[Spatial Phase, Fourier Phase, Spatial Amplitude]`) after a strict group-level 80/20 train/test split to prevent data leakage.
4. **Model**: Fine-tunes the last 2 blocks and head of `vit_tiny_patch16_224` using AdamW (`lr=2e-4`), gradient clipping, and cosine annealing.

## Results

| Test Condition | Accuracy |
| --- | --- |
| Pristine Baseline | **99.17%** |
| Spatial Crop (40% area zeroed) | **83.33%** |
| Gaussian Noise ($\sigma = 0.20$) | **53.33%** |
