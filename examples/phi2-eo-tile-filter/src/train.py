import argparse
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import accuracy_score
from .models.tiny_cnn import TinyCNN
from .utils import make_loader

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bands", type=int, default=3)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=16)
    a = p.parse_args()

    train_loader = make_loader(f"{a.data}/train", bands=a.bands, size=a.size)
    val_loader = make_loader(f"{a.data}/val", shuffle=False, bands=a.bands, size=a.size)

    model = TinyCNN(in_ch=a.bands, base=a.base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(a.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x.to(device)).argmax(1).cpu()
                ys.append(y); ps.append(pred)
        acc = accuracy_score(torch.cat(ys), torch.cat(ps))
        print(f"epoch {epoch+1} acc {acc:.3f}")

    Path("runs").mkdir(exist_ok=True)
    torch.save(model.state_dict(), Path("runs") / "tinycnn.pt")
    print("Saved runs/tinycnn.pt")

if __name__ == "__main__":
    main()
