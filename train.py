import argparse, os, json, time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
from src.data.datasets import cifar10_loaders
from src.models.cifar_effnet import build_model
from src.utils.visualize import plot_confusion_matrix
from src.utils.seed import set_seed

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])
    ap.add_argument('--data-dir', type=str, default='./data')
    ap.add_argument('--save-path', type=str, default='artifacts/model.pt')
    return ap.parse_args()

def get_device(opt):
    if opt.device == 'cpu':
        return torch.device('cpu')
    if opt.device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    all_y, all_p = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        all_y.extend(y.detach().cpu().tolist())
        all_p.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
    acc = accuracy_score(all_y, all_p)
    return sum(losses)/len(losses), acc

def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item())
            all_y.extend(y.detach().cpu().tolist())
            all_p.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
    acc = accuracy_score(all_y, all_p)
    return sum(losses)/len(losses), acc, all_y, all_p

def main():
    set_seed(42)
    opt = parse_args()
    device = get_device(opt)
    print(f"Using device: {device}")

    train_loader, test_loader, classes = cifar10_loaders(batch_size=opt.batch_size, data_dir=opt.data_dir)

    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=opt.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.epochs)

    best_acc = 0.0
    os.makedirs(os.path.dirname(opt.save_path), exist_ok=True)

    for epoch in range(1, opt.epochs+1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch:02d} | train_loss={tl:.4f} acc={ta:.4f} | val_loss={vl:.4f} acc={va:.4f}")

        # Save best
        if va > best_acc:
            best_acc = va
            torch.save({
                'model_state': model.state_dict(),
                'classes': classes,
                'arch': 'efficientnet_b0'
            }, opt.save_path)

        # Plot & save confusion matrix (watermarked)
        os.makedirs('outputs', exist_ok=True)
        plot_confusion_matrix(y_true, y_pred, classes, save_path='outputs/confusion_matrix.png')

    # Persist label map
    with open('artifacts/label_map.json', 'w') as f:
        json.dump({i: c for i, c in enumerate(classes)}, f, indent=2)

    print(f"Training done. Best val acc: {best_acc:.4f}. Model saved to {opt.save_path}.")

if __name__ == '__main__':
    main()
