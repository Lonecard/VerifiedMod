"""Procedural 16x16 textures (original pixel art, generated in code)."""
import numpy as np


def _hex(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255


def diamond_block():
    """Glowing cyan gem block: bright bevel, faceted rhombus core, sparkles."""
    t = np.zeros((16, 16, 4), np.float32)
    t[..., 3] = 1
    base, light, white, mid, dark, deep = (_hex(c) for c in
                                           ("#3fd9e6", "#9ff6ff", "#ffffff", "#25b3cc", "#13819e", "#0b5a78"))
    for y in range(16):
        for x in range(16):
            c = base
            d = abs(x - 7.5) + abs(y - 7.5)          # rhombus distance
            if d < 2:
                c = white
            elif d < 4:
                c = light if x < 8 and y < 8 else base
            elif d < 6:
                c = mid if (x > 7) or (y > 7) else base
            if x == 0 or y == 0:
                c = light
            elif x == 1 or y == 1:
                c = base
            if x in (14, 15) or y in (14, 15):
                c = dark
            if (x, y) in ((0, 15), (15, 0)):
                c = mid
            if x == 15 and y == 15:
                c = deep
            t[y, x, :3] = c
    for (x, y) in ((3, 4), (12, 11), (2, 2)):
        t[y, x, :3] = white
    return t


def pickaxe_sprite():
    """Original 16x16 cyan pickaxe: arc head + diagonal wooden handle."""
    s = np.zeros((16, 16, 4), np.float32)
    put = lambda x, y, c: s.__setitem__((y, x), np.r_[_hex(c), 1.0])
    # handle from bottom-left to the head centre (darker outline on one side)
    for i in range(10):
        x, y = 1 + i, 14 - i
        put(x, y, "#8a5a2b" if i % 3 else "#a8743c")
        if y + 1 < 16:
            put(x, y + 1, "#4a2c12")
    # curved head: points on a circle centred near the handle end
    cx, cy, r = 0.0, 15.0, 13.4
    for x in range(16):
        for y in range(16):
            dx, dy = x + 0.5 - cx, y + 0.5 - cy
            dist = np.hypot(dx, dy)
            ang = np.degrees(np.arctan2(-dy, dx))       # 45 = along the handle
            if 2 < ang < 88 and r - 1.6 < dist < r + 1.0:
                inner = dist < r - 0.4
                put(x, y, "#c9fbff" if inner else "#33d1e0")
            elif 1 < ang < 89 and r + 1.0 <= dist < r + 2.0:
                put(x, y, "#0f6f86")
    # tips
    for (x, y) in ((13, 13), (1, 1)):
        put(x, y, "#0f6f86")
    return s


def stone_block(seed=3):
    """Dark deepslate-ish stone with brick seams."""
    rng = np.random.default_rng(seed)
    t = np.ones((16, 16, 4), np.float32)
    n = rng.random((16, 16))
    g = 0.13 + 0.07 * n
    for y in (4, 5, 11, 12):
        g[y, :] *= 0.55 if y in (4, 11) else 1.25
    for x in (3, 10):
        g[0:4, x] *= 0.6
        g[12:16, x] *= 0.6
    for x in (7, 14):
        g[6:11, x] *= 0.6
    t[..., 0] = g * 0.92
    t[..., 1] = g * 0.9
    t[..., 2] = g * 1.08
    return t
