import argparse, json, os
import torch
from PIL import Image
from torchvision import transforms
from src.models.cifar_effnet import build_model
from src.utils.watermark import add_text_watermark, DEFAULT_TEXT

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--checkpoint', type=str, default='artifacts/model.pt')
    ap.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])
    ap.add_argument('--watermark', type=str, default=DEFAULT_TEXT)
    return ap.parse_args()

def get_device(opt):
    if opt.device == 'cpu':
        return torch.device('cpu')
    if opt.device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(num_classes=len(ckpt['classes']), pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt['classes']

def preprocess(img: Image.Image):
    tfm = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    return tfm(img).unsqueeze(0)

def main():
    opt = parse_args()
    device = get_device(opt)

    model, classes = load_checkpoint(opt.checkpoint, device)
    img = Image.open(opt.image).convert('RGB')
    x = preprocess(img).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(prob).item())
        cls = classes[idx]
        conf = float(prob[idx].item())

    print(json.dumps({'class': cls, 'confidence': round(conf, 4)}, indent=2))

    # Save watermarked copy with prediction text
    pred_text = f"{cls} ({conf:.2%}) - Deepak Kumar"
    wm = add_text_watermark(img, text=pred_text)
    os.makedirs('outputs', exist_ok=True)
    base = os.path.splitext(os.path.basename(opt.image))[0]
    out_path = f'outputs/{base}_pred.jpg'
    wm.save(out_path)
    print(f"Saved watermarked prediction image to {out_path}")

if __name__ == '__main__':
    main()
