# 🧱 Autotile — Bitwise Wang Tiling — raylib + pyray

A **bitwise autotiling** demo built with Python, using an **8-bit bitmask** per tile to automatically select the correct sprite variant based on neighbouring walls — generated on a procedural cave map and editable in real time with **raylib** and **pyray**.

---

## ✨ Features

* **8-bit bitmask autotiling** — each wall tile checks all 8 neighbours and encodes them into a single integer to look up the correct sprite variant
* **Diagonal normalisation** — hanging diagonal bits are stripped when the shared cardinal neighbours are absent (e.g. a corner tile is only used when both adjacent edge tiles are also walls), keeping the tileset visually consistent
* **Bitmask deduplication** — all 256 raw bitmask values are collapsed into a compact mapping of only the unique normalised variants, so the spritesheet stays small
* **Vectorised bitmask computation** — `compute_bitmask` uses `scipy.ndimage.correlate` with the 8-directional bit kernel to compute the full grid's bitmask array in one pass
* **Procedural cave generation** — hash-based integer noise seeded with `SEED`, smoothed with configurable cellular automata passes into natural cave shapes
* **Real-time tile editing** — left / right mouse places and destroys tiles; only the 3×3 neighbourhood around the edited tile has its bitmask recomputed, leaving the rest of the grid untouched

---

## 📦 Requirements

Make sure you have Python installed, then install dependencies:

```bash
pip install raylib numpy scipy
```

---

## 🚀 Running the Project

```bash
python main.py
```

---

## 🎮 Controls

| Key / Button        | Action               |
| ------------------- | -------------------- |
| Left Mouse Button   | Place tile (wall)    |
| Right Mouse Button  | Destroy tile (empty) |
| ESC / Close Window  | Exit program         |

---

## ⚙️ Configuration

Key constants at the top of `main.py`:

```python
SEED = 534543          # World generation seed
TILE_SIZE = 48         # Rendered tile size in pixels
SMOOTHING_AMOUNT = 8   # Cellular automata passes — higher = rounder caves
CHUNK_PADDING = 4      # Border padding for edge smoothing
```

Grid size is set in `Main.__init__`:

```python
self.width  = 20   # Tile columns
self.height = 20   # Tile rows
```

---

## 📖 References

* [How to Use Tile Bitmasking to Auto-Tile Your Level Layouts](https://code.tutsplus.com/how-to-use-tile-bitmasking-to-auto-tile-your-level-layouts--cms-25673t) — Tuts+ tutorial used as the foundation for the bitmask logic

---

## 🖼️ Preview

![App Screenshot](screenshot.gif)
