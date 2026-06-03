# 🎨 TensorFlow Neural Style Transfer

A modular, production-ready implementation of **Neural Style Transfer** and **image generation** using TensorFlow 2 / Keras. Blend the content of any photo with the artistic style of any painting — in minutes.

> Based on the seminal paper: *"A Neural Algorithm of Artistic Style"* — Gatys et al. (2015)

---

## ✨ Features

- 🖼️ **Neural Style Transfer** — fuse content & style images using VGG19 deep features
- ⚡ **Fast Style Transfer** — train a feed-forward network for real-time stylization
- 🧱 **Modular Keras architecture** — clean custom layers, loss functions, and training loops
- ⚙️ **YAML config** — full experiment control without touching code
- 📊 **TensorBoard integration** — live loss curves and image previews
- 💾 **Checkpoint & resume** — never lose a long stylization run
- 🖥️ **CLI + Python API** — use as a script or import as a library
- 🧪 **Full test suite** — pytest-ready unit tests

---

## 📁 Project Structure

```
tf-neural-style/
├── configs/
│   └── default.yaml          # Experiment configuration
├── src/
│   ├── model.py              # VGG19 feature extractor + Fast Style Net
│   ├── losses.py             # Content, style (Gram matrix), total variation loss
│   ├── stylizer.py           # Optimization-based style transfer engine
│   ├── fast_stylizer.py      # Feed-forward fast style transfer trainer
│   ├── preprocessing.py      # Image loading, resizing, normalization
│   └── utils.py              # Config, logging, callbacks, helpers
├── scripts/
│   ├── stylize.py            # Run optimization-based style transfer
│   ├── train_fast.py         # Train a fast style network
│   └── apply_fast.py         # Apply a trained fast style network
├── tests/
│   └── test_losses.py        # Unit tests for loss functions & model shapes
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run Style Transfer (optimization-based)

```bash
python scripts/stylize.py \
  --content assets/content.jpg \
  --style   assets/style.jpg \
  --output  outputs/result.jpg
```

### 3. Train a Fast Style Network

```bash
python scripts/train_fast.py \
  --style assets/starry_night.jpg \
  --dataset path/to/coco/train2017/
```

### 4. Apply Fast Style Network

```bash
python scripts/apply_fast.py \
  --checkpoint outputs/fast_model/ \
  --image path/to/photo.jpg \
  --output outputs/styled.jpg
```

---

## 🎛️ Configuration

Edit `configs/default.yaml` to control everything:

```yaml
style_transfer:
  image_size: 512
  iterations: 1000
  content_weight: 1.0e4
  style_weight: 1.0e-2
  tv_weight: 30.0
```

---

## 📊 How It Works

```
Content Image ──► VGG19 ──► Content Features ──► Content Loss ──┐
                                                                  ├──► Total Loss ──► Adam ──► Stylized Image
Style Image ───► VGG19 ──► Gram Matrices ────► Style Loss ──────┘
                                               TV Regularization ┘
```

The algorithm iteratively updates pixel values of a generated image to minimize:
- **Content loss** — preserve high-level structure from content image
- **Style loss** — match Gram matrix statistics from style image
- **Total variation loss** — encourage spatial smoothness

---

## 🔧 Models

| Mode | Speed | Quality | Use Case |
|---|---|---|---|
| Optimization (VGG19) | ~5 min/image | ⭐⭐⭐⭐⭐ | Best quality, slow |
| Fast Style Net | ~0.01s/image | ⭐⭐⭐⭐ | Real-time, per-style training |

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📄 License

MIT — free to use and modify.

---

## 📚 References

- Gatys et al. (2015) — [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- Johnson et al. (2016) — [Perceptual Losses for Real-Time Style Transfer](https://arxiv.org/abs/1603.08155)
