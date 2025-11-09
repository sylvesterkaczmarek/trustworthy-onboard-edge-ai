from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TileFolder(Dataset):
    def __init__(self, root: str, bands: int = 3, size: int = 64):
        self.root = Path(root)
        self.paths = []
        for cls, name in enumerate(["background", "event"]):
            for p in sorted((self.root / name).glob("*.png")):
                self.paths.append((p, cls))
        self.bands = bands
        self.size = size

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p, cls = self.paths[idx]
        img = Image.open(p).convert("RGB").resize((self.size, self.size))
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        y = torch.tensor(cls, dtype=torch.long)
        return x, y

def make_loader(path: str, batch: int = 64, shuffle: bool = True, bands: int = 3, size: int = 64):
    ds = TileFolder(path, bands=bands, size=size)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0)
