# Data

Synthetic tiles used by the demo. Two classes: `background` and `event`.

## Generate
```bash
# from examples/phi2-eo-tile-filter
python -m data.synth --out ./tiles --n 200 --bands 3 --size 64
```
- `--n` total tiles split 80% train / 20% val  
- `--bands` input channels (use 3 here)  
- `--size` square tile size in pixels

## Output layout
```
tiles/
├─ train/
│  ├─ background/*.png
│  └─ event/*.png
└─ val/
   ├─ background/*.png
   └─ event/*.png
```
## Using your own EO tiles
Place PNGs under the same folder structure as above. Any size works; scripts resize to the `--size` you pass. Keep at least ~50 images per class in `val` so INT8 calibration has enough samples.

## Tips
- Delete and re-generate tiles to refresh the dataset.
- Keep real data out of git; the repo `.gitignore` excludes `tiles/`.
