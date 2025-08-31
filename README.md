# Advanced AI/ML Project – CIFAR-10 Classifier + API (Watermark: Deepak Kumar)

An end-to-end deep learning project you can push to GitHub right away. It includes:

- PyTorch training pipeline (EfficientNet-B0 transfer learning on CIFAR-10)
- Evaluation + confusion matrix
- Inference CLI (`inference.py`) with **automatic watermark** on outputs
- Production-ready **FastAPI** service with image upload + prediction
- Dockerfile for containerized deployment
- Tests, modular code, and clean repo structure

> All generated images/plots are watermarked with **"Deepak Kumar"** by default.

## Repo Structure

```
ai-ml-advanced-project-watermark-deepak-kumar/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── Dockerfile
├── train.py
├── inference.py
├── api/
│   └── main.py
├── src/
│   ├── data/datasets.py
│   ├── models/cifar_effnet.py
│   └── utils/{
│         watermark.py, visualize.py, seed.py
│       }
├── tests/test_watermark.py
├── artifacts/            # saved models, label maps
├── outputs/              # predictions, plots (watermarked)
└── notebooks/quickstart.ipynb (optional starter notebook)
```

## Quickstart

### 1) Create & activate environment
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows) .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Train (downloads CIFAR-10 automatically)
```bash
python train.py --epochs 5 --batch-size 128 --lr 3e-4 --device auto
```
Artifacts:
- `artifacts/model.pt`
- `artifacts/label_map.json`
- `outputs/confusion_matrix.png` (watermarked)

### 3) Inference (CLI)
```bash
python inference.py --image path/to/image.jpg --checkpoint artifacts/model.pt --device auto
```
Outputs prediction to console and saves **watermarked** image to `outputs/`.

### 4) Run API (FastAPI)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Open `http://localhost:8000/docs` to try the `/predict` endpoint with an image.
- Response includes top-1 class + probability. A **watermarked** copy is saved in `outputs/`.

### 5) Docker (optional)
```bash
docker build -t cifar-api .
docker run -p 8000:8000 cifar-api
```

### 6) Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: Advanced AI/ML project with watermarking"
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

## Notes
- Default watermark text: `"Deepak Kumar"` (change in `src/utils/watermark.py` if needed).
- If you don't have a GPU, set `--device cpu`. `--device auto` picks CUDA if available.
- CIFAR-10 is used for quick experimentation; swap with your dataset by editing `src/data/datasets.py`.
