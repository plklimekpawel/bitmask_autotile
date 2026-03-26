from os.path import join

from raylib import *
from pyray import *
import numpy as np
from scipy.ndimage import correlate

SEED = 534543
TILE_SIZE = 48
LIGHT_GRAY = (175, 175, 175, 255)
DARK_GRAY = (25, 25, 25, 255)
WINDOW_WIDTH, WINDOW_HEIGHT = 960, 960
CHUNK_PADDING = 4
SMOOTHING_AMOUNT = 8
NEIGHBOUR_KERNEL = np.array([[1,1,1],
                                    [1,0,1],
                                    [1,1,1]], dtype=np.uint8)
BIT_KERNEL = np.array([
    [1,   2,  4],
    [8,   0, 16],
    [32, 64, 128]
], dtype=np.uint8)

def import_spritesheet(tile_size, *path):
    path = join(*path)
    spritesheet = load_texture(path + '.png')

    data = []
    width, height = spritesheet.width // tile_size, spritesheet.height // tile_size
    for y in range(height):
        for x in range(width):
            source = Rectangle(x * tile_size, y * tile_size, tile_size, tile_size)
            data.append(source)
    return data, spritesheet

def noise_wall(row, col, threshold=0.45):
    # 0xFFFFFFFF -> 32 bit mask
    # * and + is always vectorized in numpy
    h = (row * 374761393 + col * 668265263 + SEED) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177
    h = h & 0xFFFFFFFF
    return (h / 2**32) < threshold

def compute_bitmask(tiles):
    mask = correlate(tiles, BIT_KERNEL, mode='constant', cval=1)

    mask[((mask & 2) == 0) | ((mask & 8) == 0)] &= 0b11111110
    mask[((mask & 2) == 0) | ((mask & 16) == 0)] &= 0b11111011
    mask[((mask & 64) == 0) | ((mask & 8) == 0)] &= 0b11011111
    mask[((mask & 64) == 0) | ((mask & 16) == 0)] &= 0b01111111

    return mask

def compute_bitmask_tile(tiles, y, x):
    height, width = tiles.shape
    def get(y, x):
        if 0 <= y < height and 0 <= x < width:
            return tiles[y, x]
        return 1

    mask = (
        (get(y - 1, x - 1)  << 0) |
        (get(y - 1, x)      << 1) |
        (get(y - 1, x + 1)  << 2) |
        (get(y, x - 1)      << 3) |
        (get(y, x + 1)      << 4) |
        (get(y + 1, x - 1)  << 5) |
        (get(y + 1, x)      << 6) |
        (get(y + 1, x + 1)  << 7)
    )
    mask = normalize(mask)
    return mask

def normalize(bitmask):
    # 1  | 2  | 4
    # 8  | -  | 16
    # 32 | 64 | 128

    # & -> AND - 1 only when both bits are 1
    # 0b00000111 = 7    (bitmask)
    # 0b00000010 = 2
    # ----------
    # 0b00000010 = 2 - True, bit 2 is on in bitmask with value 7

    # ~ -> NOT - 1 turns into 0 and 0 into 1
    # 0b00000001 -> 0b11111110

    # 0b00000111 = 7    (bitmask)
    # 0b11111110 = ~1
    # ----------
    # 0b00000110 = 6 - Turns off bit 1

    # 0b00000010 = 2    (bitmask)
    # 0b11111110 = ~1
    # ----------
    # 0b00000010 = 2 - Nothing changes because bit 1 is already off
    if not (bitmask & 2) or not (bitmask & 8):  bitmask &= ~1    # If no tile above(0, 1)[2-bit] or left(-1, 0)[8-bit] then get rid of diagonal topleft(-1, -1)[1-bit]
    if not (bitmask & 2) or not (bitmask & 16): bitmask &= ~4    # If no tile above(0, 1)[2-bit] or right(1, 0)[16-bit] then get rid of diagonal topright(1, -1)[4-bit]
    if not (bitmask & 64) or not (bitmask & 8): bitmask &= ~32   # If no tile down(0, -1)[64-bit] or left(-1, 0)[8-bit] then get rid of diagonal downleft(-1, 1)[32-bit]
    if not (bitmask & 64) or not (bitmask & 16):bitmask &= ~128  # If no down above(0, -1)[64-bit] or right(1, 0)[16-bit] then get rid of diagonal downright(1, 1)[128-bit]
    return bitmask

def generate_bitmask_mapping():
    bitmask_mapping = {}
    for bitmask in range(256):
        normalized_bitmask = normalize(bitmask) # Gets rid of hanging diagonals
        if normalized_bitmask not in bitmask_mapping:
            bitmask_mapping[normalized_bitmask] = len(bitmask_mapping)
    return bitmask_mapping

class Main:
    def __init__(self):
        init_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Autotile")
        set_target_fps(120)

        self.tiles = None
        self.bitmask = None
        self.bitmask_mapping = generate_bitmask_mapping()

        self.width = 20
        self.height = 20

        self.data, self.spritesheet = import_spritesheet(16, join('47-tiles-16x16-autotile-bit-wise'))

        self._generate_chunk()

    def _get_tile_index(self, y, x):
        return self.bitmask_mapping[int(self.bitmask[y, x])]

    def _generate_chunk(self):
        rows = np.arange(self.width)[:, None]   # Ys
        cols = np.arange(self.height)[None, :]  # Xs
        tiles = noise_wall(rows, cols).astype(np.uint8)

        for _ in range(SMOOTHING_AMOUNT):
            tiles = self._smooth_step(tiles.astype(np.uint8))

        self.tiles = tiles
        self.bitmask = compute_bitmask(self.tiles)

    def _smooth_step(self, tiles):
        neighbors_count, solid_mask = self._neighbours_count_array(tiles)
                 # where(condition, value_if_true, value_if_false)
        new_tiles = np.where(solid_mask, neighbors_count >= 4, neighbors_count >= 5)
        result = np.where(new_tiles, 1, 0)

        return result

    def _neighbours_count_array(self, tiles):
        solid_mask = np.where(tiles >= 1, 1, 0).astype(np.uint8)
        neighbors_count = correlate(solid_mask, NEIGHBOUR_KERNEL, mode='constant', cval=1)
        return neighbors_count, solid_mask

    def _update_tile_bitmask(self, y, x):
        for offset_y in (-1, 0, 1):
            for offset_x in (-1, 0, 1):
                ny, nx = y + offset_y, x + offset_x
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    self.bitmask[ny, nx] = compute_bitmask_tile(self.tiles, ny, nx)

    def _set_tile(self, y, x, value):
        if self.tiles[y, x] == value: return

        self.tiles[y, x] = value
        self._update_tile_bitmask(y, x)

    def update(self):
        mouse_pos = get_mouse_position()
        tile_x, tile_y = int(mouse_pos.x // TILE_SIZE), int(mouse_pos.y // TILE_SIZE)
        if is_mouse_button_down(0):
            self._set_tile(tile_y, tile_x, 1)
        if is_mouse_button_down(1):
            self._set_tile(tile_y, tile_x, 0)

    def draw(self):
        begin_drawing()
        clear_background(LIGHT_GRAY)

        walls = np.argwhere(self.tiles != 0)
        for y, x in walls:  # numpy is indexed [y, x]
            pixel_x, pixel_y = int(x * TILE_SIZE), int(y * TILE_SIZE)
            tile_id = self._get_tile_index(y, x)
            draw_texture_pro(self.spritesheet, self.data[tile_id], Rectangle(pixel_x, pixel_y, TILE_SIZE, TILE_SIZE), Vector2(), 0, WHITE)

        draw_fps(10, get_screen_height() - 20)
        end_drawing()

    def run(self):
        while not window_should_close():
            self.update()
            self.draw()
        close_window()

if __name__ == '__main__':
    game = Main()
    game.run()
