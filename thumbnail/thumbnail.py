"""
EVERYTHING? - YouTube thumbnail (1280x720).

Run:  python3 thumbnail.py
Outputs (in ./out):
    thumbnail.png          final 1280x720 PNG
    thumbnail_400.png      400px preview (phone-size check)
    layers/*.png           every layer at full 2x working resolution
    thumbnail_layers.ora   layered file (open in Krita / GIMP / Paint.NET plugin)

Everything you might want to tweak lives in CONFIG below: positions are in
final 1280x720 pixel coordinates, poses are joint angles in degrees.
"""
import math
import os
import zipfile
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops

import mcrender as mc
import assets

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

CONFIG = dict(
    size=(1280, 720),
    ss=int(os.environ.get("SS", 2)),                                   # composite at 2x, downsample at the end
    text=dict(enabled=False,                 # the references carry no title text
              string="EVERYTHING?", xy=(46, 26), width=640, rotate=-4,
              fill_top="#fff9d6", fill_mid="#ffd84a", fill_bot="#ff8a1a",
              outline="#12021c", outline_px=11, extrude=16),
    raven=dict(                              # background: on a ledge in the lava chamber
        pos=(120, 48, -152), yaw=-36, ledge=3,
        pose=dict(head=(6, 0, 0), body=(0, 0, 0), arm_r=(-84, 18, 0),
                  arm_l=(-4, 0, 10), leg_r=(0, 0, 2), leg_l=(0, 0, -2)),
        rim_left="#ff4fa0", rim_right="#ffb23f"),
    cave=dict(                               # hero shot: Lonecard + diamond ore stack in a cave
        cam_pos=(-2, 25, 54), cam_target=(30, 23, -80), fov=40,
        stack_pos=(36, 0, -26), stack_yaw=-20, ore_glow=0.95,
        root_yaw=24, pick_roll=90,
        pose=dict(head=(-4, -4, 0), body=(0, 4, 0), arm_r=(-140, 0, -20),
                  arm_l=(-24, 0, -10), leg_r=(14, 0, 0), leg_l=(-12, 0, 0)),
        rim_left="#c04dff", rim_right="#5ff4ff"),
    diamond_glow=dict(center=(880, 470), radius=320, color="#22e8ff"),
    lava_glow=dict(center=(1080, 200), radius=380),
)


# --------------------------------------------------------------------------- #
# small image helpers (float RGB arrays in 0..1)
# --------------------------------------------------------------------------- #
def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255


def screen(a, b):
    return 1 - (1 - a) * (1 - b)


def _box(a, rad, axis):
    if rad < 1:
        return a
    pad = [(0, 0), (0, 0)]
    pad[axis] = (rad + 1, rad)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * rad + 1, 2 * rad + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * rad + 1)).astype(np.float32)


def _gauss(a, sigma):
    """Gaussian approximated by 3 box passes per axis."""
    rad = max(1, int(round((math.sqrt(12 * sigma * sigma / 3 + 1) - 1) / 2)))
    for _ in range(3):
        a = _box(_box(a, rad, 0), rad, 1)
    return a


def blur(arr, r):
    """Gaussian blur of a float HxW or HxWxC array (done at reduced res for big radii)."""
    if r <= 0:
        return arr
    f = 1
    while r / f > 24 and f < 8:
        f *= 2
    h, w = arr.shape[:2]
    chans = [arr] if arr.ndim == 2 else [arr[..., i] for i in range(arr.shape[2])]
    out = []
    for c in chans:
        im = Image.fromarray(c.astype(np.float32), "F")
        if f > 1:
            im = im.resize((max(1, w // f), max(1, h // f)), Image.BILINEAR)
        a = _gauss(np.asarray(im, np.float32), r / f)
        if f > 1:
            a = np.asarray(Image.fromarray(a, "F").resize((w, h), Image.BILINEAR), np.float32)
        out.append(a)
    return out[0] if arr.ndim == 2 else np.dstack(out)


def shift(m, dx, dy):
    out = np.zeros_like(m)
    H, W = m.shape
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    out[yd, xd] = m[ys, xs]
    return out


def radial(W, H, cx, cy, r, power=2.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    return np.exp(-d ** power)


# --------------------------------------------------------------------------- #
# 3-D layers
# --------------------------------------------------------------------------- #
def render_layer(faces, cam, lights, height_px, render_px=1400):
    """Render, crop to content, scale (down) to the requested pixel height."""
    # probe render to measure the subject, then render big enough to only downscale
    probe = mc.render(faces, cam, 300, 300, lights)["rgba"][..., 3]
    ys = np.nonzero(probe.any(1))[0]
    frac = (ys.max() - ys.min() + 1) / 300
    render_px = int(math.ceil(height_px / frac * 1.08))
    r = mc.render(faces, cam, render_px, render_px, lights)
    a = r["rgba"][..., 3]
    ys, xs = np.nonzero(a)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = lambda arr: arr[y0:y1, x0:x1]
    scale = height_px / (y1 - y0)
    size = (max(1, round((x1 - x0) * scale)), height_px)
    if scale > 1.0:
        print(f"  warning: layer upscaled x{scale:.2f} - raise render_px for sharper pixels")
    rgba = Image.fromarray((crop(r["rgba"]) * 255).astype(np.uint8)).resize(size, Image.LANCZOS)
    emis = Image.fromarray((np.clip(crop(r["emis"]), 0, 1) * 255).astype(np.uint8)).resize(size, Image.LANCZOS)
    tags = {}
    for i, t in enumerate(r["tags"]):
        m = Image.fromarray(((crop(r["tag"]) == i) * 255).astype(np.uint8)).resize(size, Image.LANCZOS)
        tags[t] = np.asarray(m, np.float32) / 255
    return dict(rgba=np.asarray(rgba, np.float32) / 255, emis=np.asarray(emis, np.float32) / 255, tags=tags)


def build_raven(cfg, ss):
    skin = mc.load_tex(os.path.join(HERE, "skins", "raven_9qmm.png"))
    root = mc.Xform(mc.rot(0, cfg["root_yaw"], 0))
    faces, _ = mc.player_faces(skin, root, cfg["pose"], tag="raven")
    lights = mc.Lights(
        ambient=(0.55, 0.45, 0.6),
        key=((-0.4, 0.6, 1.0), (0.9, 0.85, 0.95)),
        rims=[((1.0, 0.25, -1.0), hexc("#ff4f9a") * 0.4, 2.0),
              ((-1.0, 0.3, -1.0), hexc(cfg["rim_left"]) * 0.3, 2.0),
              ((0, 1, 0), hexc("#ff9a5a") * 0.10, 1.0)],
        sheen=((-0.5, 0.35, 1.0), hexc("#7a5cff") * 0.16))
    cam = mc.Camera((0, 20, 0), yaw=cfg["yaw"], pitch=cfg["pitch"], dist=cfg["dist"], fov=cfg["fov"])
    return render_layer(faces, cam, lights, cfg["height"] * ss, render_px=1900)


def build_cave(cfg, ss, W, H):
    """One 3-D scene: cave + Lonecard + pickaxe + diamond-ore stack, full frame."""
    rng = np.random.default_rng(cfg.get("seed", 4))
    skin = mc.load_tex(os.path.join(HERE, "skins", "lonecard.png"))
    pick = assets.pickaxe_sprite()
    root = mc.Xform(mc.rot(0, cfg["root_yaw"], 0))
    faces, joints = mc.player_faces(skin, root, cfg["pose"], tag="lonecard")

    # pickaxe in the right hand: sprite handle axis (1,1) -> along the arm (-y),
    # head arc (-1,1) -> arm forward (+z)
    s = 1.05
    R = np.column_stack([np.array([0, -1, -1]) / math.sqrt(2),
                         np.array([0, -1, 1]) / math.sqrt(2),
                         np.array([-1, 0, 0])])
    grip = np.array([3.5, 3.5, 0.0])
    hand = np.array([0, -8.5, 0.0])
    roll = mc.Xform(mc.rot(0, cfg.get("pick_roll", 0), 0))      # spin the tool around the arm axis
    item = joints["arm_r"] @ roll @ mc.Xform(R, hand - R @ (s * grip), s)
    faces += mc.sprite_voxels(pick, item, "pickaxe", depth=1.0, emissive=0.5)

    # diamond ore stack (2 + 1) set into a small stone outcrop
    w = mc.Xform(mc.rot(0, cfg["stack_yaw"], 0), cfg["stack_pos"])
    for n, (x, y, z) in enumerate([(-16, 0, -8), (0, 0, -8), (-8, 16, -8)]):
        faces += mc.block_faces(assets.diamond_ore(n), x, y, z, w, "ore", emissive=cfg["ore_glow"])
    for n, (x, y, z) in enumerate([(16, 0, -24), (-16, 0, -24), (0, 0, -24),
                                   (-8, 16, -24), (8, 16, -24)]):
        faces += mc.block_faces(assets.cave_stone(n % 3), x, y, z, w, "world")

    # --- cave volume on a 16-unit grid
    tex = {f"s{n}": assets.cave_stone(n) for n in range(4)}
    tex.update({f"d{n}": assets.cave_deepslate(n) for n in range(3)})
    tex.update({"lava0": assets.lava(0), "lava1": assets.lava(1), "ore": assets.diamond_ore(9)})
    stone = lambda: f"s{rng.integers(0, 4)}"
    slate = lambda: f"d{rng.integers(0, 3)}"
    cells = {}
    for i in range(-12, 24):
        for k in range(-18, 8):
            cells[(i, -1, k)] = slate() if rng.random() < 0.6 else stone()
            cells[(i, 11, k)] = stone()                         # ceiling
            if rng.random() < 0.18 and k < 2:                   # stalactites
                for jj in range(10, 10 - rng.integers(1, 4), -1):
                    cells[(i, jj, k)] = stone()
    # near back wall (behind Lonecard) with uneven face
    for i in range(-12, 6):
        for j in range(0, 11):
            if i >= 3 and j < 6 + (i - 3) * 2:                  # opening into the lava chamber
                continue
            cells[(i, j, -4)] = stone() if rng.random() < 0.7 else slate()
            if j >= 3 and rng.random() < 0.35:
                cells[(i, j, -3)] = stone()
    for j in range(0, 11):                                      # left wall
        for k in range(-4, 8):
            cells[(-12, j, k)] = stone()
    # deep chamber to the right: far wall, lava pool, lava fall, rock columns
    for i in range(6, 24):
        for j in range(0, 11):
            cells[(i, j, -18)] = slate() if rng.random() < 0.5 else stone()
    for j in range(0, 11):
        for k in range(-18, 8):
            cells[(23, j, k)] = stone()
    for i in range(9, 16):
        for k in range(-14, -8):
            if not ((i in (9, 15)) and (k in (-14, -9))):
                cells[(i, -1, k)] = "lava0" if (i + k) % 2 else "lava1"
    for j in range(-1, 11):
        cells[(8, j, -17)] = "lava0" if j % 2 else "lava1"
        cells[(16, j, -17)] = "lava1" if j % 2 else "lava0"
    for (i, k, h) in [(19, -10, 4), (13, -4, 1), (21, -3, 3), (11, -15, 2)]:
        for j in range(h):
            cells[(i, j, k)] = stone()
    for (i, k) in [(-4, 3), (-7, -1), (8, 2)]:                    # loose rocks on the floor
        cells[(i, 0, k)] = stone()
    faces += mc.voxel_faces(cells, mc.Xform(), textures=tex,
                            tag_of=lambda t: "lava" if t.startswith("lava") else "world",
                            emissive_of=lambda t: 0.9 if t.startswith("lava") else (0.5 if t == "ore" else 0.0))

    # Raven, small, in the background on a rock ledge, pointing at Lonecard
    rv = CONFIG["raven"]
    rskin = mc.load_tex(os.path.join(HERE, "skins", "raven_9qmm.png"))
    rf, _ = mc.player_faces(rskin, mc.Xform(mc.rot(0, rv["yaw"], 0), rv["pos"]), rv["pose"], tag="raven")
    faces += rf
    li, lk = int(rv["pos"][0] // 16), int(rv["pos"][2] // 16)
    ledge = {}
    for di in (-1, 0, 1):
        for dk in (-1, 0, 1):
            for j in range(rv["ledge"] - (1 if abs(di) + abs(dk) == 2 else 0)):
                ledge[(li + di, j, lk + dk)] = assets.cave_deepslate((di + dk) % 3)
    faces += mc.voxel_faces(ledge, mc.Xform(), tag_of=lambda t: "world")

    ore_c = np.asarray(cfg["stack_pos"], float) + np.array([-2, 16, 6])
    lights = mc.Lights(
        ambient=(0.11, 0.09, 0.17),
        key=((-0.4, 0.8, 0.6), (0.22, 0.22, 0.36)),
        rims=[((1.0, 0.2, -0.8), hexc(cfg["rim_right"]) * 0.3, 2.0),
              ((-1.0, 0.4, -0.8), hexc(cfg["rim_left"]) * 0.25, 2.0)],
        points=[(ore_c + np.array([-18, 6, 22]), hexc("#3feaff") * 1.6, 30),
                ((200, 10, -180), hexc("#ff6a1a") * 2.6, 95),
                ((136, 90, -250), hexc("#ff7a22") * 2.4, 110),
                ((264, 90, -250), hexc("#ff7a22") * 1.6, 100),
                ((-120, 120, 40), hexc("#6a4cff") * 0.5, 160)],
        sheen=((-0.3, 0.4, 1.0), hexc("#7a5cff") * 0.13),
        fog=(hexc("#0b0614"), 1 / 700))
    cam = mc.Camera.look(cfg["cam_pos"], cfg["cam_target"], fov=cfg["fov"])
    r = mc.render(faces, cam, W, H, lights)
    tags = {t: (r["tag"] == i).astype(np.float32) for i, t in enumerate(r["tags"])}
    return dict(rgba=r["rgba"], emis=r["emis"], tags=tags, depth=r["depth"])


# --------------------------------------------------------------------------- #
# 2-D compositing pieces
# --------------------------------------------------------------------------- #
def background(W, H, ss):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = yy / H
    u = xx / W
    top, bot = hexc("#14042c"), hexc("#3b0d63")
    bg = top * (1 - v[..., None]) + bot * v[..., None]
    bg = bg * (0.85 + 0.25 * u[..., None])          # a touch brighter to the right

    bx, by = (c * ss for c in CONFIG["burst"]["center"])
    # god rays from the burst
    ang = np.arctan2(yy - by, xx - bx)
    rays = (0.5 + 0.5 * np.cos(ang * 9 + 0.6)) ** 6 + 0.6 * (0.5 + 0.5 * np.cos(ang * 23 + 2.0)) ** 10
    far = radial(W, H, bx, by, 760 * ss, 1.3)
    bg = screen(bg, hexc("#b0157a")[None, None] * (radial(W, H, bx, by, 560 * ss, 1.6) * 0.85)[..., None])
    bg = screen(bg, hexc("#ff6a1a")[None, None] * (radial(W, H, bx, by, 330 * ss, 1.8) * 0.95)[..., None])
    bg = screen(bg, hexc("#ffd27a")[None, None] * (radial(W, H, bx, by, 150 * ss, 2.0) * 0.9)[..., None])
    bg = screen(bg, hexc("#ff8a3d")[None, None] * (rays * far * 0.35)[..., None])

    # orange ember haze along the bottom-right horizon, purple deep in the top-left
    bg = screen(bg, hexc("#ff5a1f")[None, None] * (radial(W, H, 1150 * ss, 760 * ss, 380 * ss, 2) * 0.55)[..., None])

    dg = CONFIG["diamond_glow"]
    gx, gy = (c * ss for c in dg["center"])
    bg = screen(bg, hexc(dg["color"])[None, None] * (radial(W, H, gx, gy, dg["radius"] * ss, 1.6) * 0.75)[..., None])
    bg = screen(bg, hexc("#e8ffff")[None, None] * (radial(W, H, gx, gy, 90 * ss, 2) * 0.55)[..., None])

    # soft block-pixel texture so the gradient isn't flat (big faint squares)
    rng = np.random.default_rng(7)
    cell = 40 * ss
    grid = rng.random((H // cell + 1, W // cell + 1)).astype(np.float32)
    grid = np.kron(grid, np.ones((cell, cell), np.float32))[:H, :W]
    bg *= (0.97 + 0.04 * blur(grid, 2 * ss))[..., None]
    return np.clip(bg, 0, 1)


def place(canvas_shape, layer, anchor, ss):
    """Return full-canvas arrays (rgba, emis, tag masks) with the layer centred on anchor."""
    H, W = canvas_shape
    lh, lw = layer["rgba"].shape[:2]
    cx, cy = anchor[0] * ss, anchor[1] * ss
    x0, y0 = int(round(cx - lw / 2)), int(round(cy - lh / 2))

    def paste(arr):
        out = np.zeros((H, W) + arr.shape[2:], np.float32)
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        w = min(lw - sx0, W - dx0)
        h = min(lh - sy0, H - dy0)
        if w > 0 and h > 0:
            out[dy0:dy0 + h, dx0:dx0 + w] = arr[sy0:sy0 + h, sx0:sx0 + w]
        return out

    return dict(rgba=paste(layer["rgba"]), emis=paste(layer["emis"]),
                tags={k: paste(v) for k, v in layer["tags"].items()})


def rim(mask, dx, dy, width, color, strength):
    """Lit edge on the side of the silhouette facing (dx, dy)."""
    m = mask
    edge = np.zeros_like(m)
    for k in range(1, width + 1):
        edge = np.maximum(edge, m * (1 - shift(m, -int(dx * k), -int(dy * k))))
    edge = blur(edge, 1.2) * m
    return color[None, None] * (edge * strength)[..., None]


def over(dst, rgb, a):
    return dst * (1 - a[..., None]) + rgb * a[..., None]


def sparkles(W, H, ss, centers, rng):
    """4-point pixel sparkles + square particles. Returns additive RGB."""
    layer = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(layer)
    for (cx, cy, spread, n, col, smin, smax) in centers:
        for _ in range(n):
            x = cx + rng.normal(0, spread)
            y = cy + rng.normal(0, spread * 0.7)
            s = rng.uniform(smin, smax) * ss
            c = tuple(int(v * 255) for v in hexc(col))
            if rng.random() < 0.45:
                d.rectangle([x * ss - s, y * ss - s / 4, x * ss + s, y * ss + s / 4], fill=c)
                d.rectangle([x * ss - s / 4, y * ss - s, x * ss + s / 4, y * ss + s], fill=c)
            else:
                q = s * 0.45
                d.rectangle([x * ss - q, y * ss - q, x * ss + q, y * ss + q], fill=c)
    arr = np.asarray(layer, np.float32) / 255
    return screen(arr, blur(arr, 6 * ss) * 1.6)


def text_layer(W, H, ss, cfg):
    font_path = os.path.join(HERE, "fonts", "LuckiestGuy.ttf")
    target_w = cfg["width"] * ss
    size = 100
    font = ImageFont.truetype(font_path, size)
    bbox = font.getbbox(cfg["string"])
    size = int(size * target_w / (bbox[2] - bbox[0]))
    font = ImageFont.truetype(font_path, size)
    bbox = font.getbbox(cfg["string"])
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 80 * ss
    cw, ch = tw + 2 * pad, th + 2 * pad
    org = (pad - bbox[0], pad - bbox[1])

    def mask(stroke=0, off=(0, 0)):
        im = Image.new("L", (cw, ch))
        ImageDraw.Draw(im).text((org[0] + off[0], org[1] + off[1]), cfg["string"], font=font, fill=255,
                                stroke_width=stroke, stroke_fill=255)
        return im

    glyph = mask()
    ow = cfg["outline_px"] * ss
    # 3-D extrusion (down-right), darker purple, then thick outline around everything
    ext = Image.new("L", (cw, ch))
    steps = cfg["extrude"] * ss // 2
    for k in range(1, steps + 1):
        ext = ImageChops.lighter(ext, mask(0, (round(k * 0.55), k)))
    outline = ImageChops.lighter(mask(ow), ext.filter(ImageFilter.MaxFilter(2 * (ow // 2) + 1)))

    g = np.asarray(glyph, np.float32) / 255
    e = np.asarray(ext, np.float32) / 255
    o = np.asarray(outline, np.float32) / 255

    yy = np.linspace(0, 1, ch, dtype=np.float32)[:, None]
    t0 = (yy - org[1] / ch).clip(0, 1)
    # vertical fill gradient restricted to glyph box
    gy = ((np.arange(ch)[:, None] - (pad)) / th).clip(0, 1).astype(np.float32)
    top, mid, bot = hexc(cfg["fill_top"]), hexc(cfg["fill_mid"]), hexc(cfg["fill_bot"])
    fill = np.where(gy[..., None] < 0.45, top + (mid - top) * (gy[..., None] / 0.45),
                    mid + (bot - mid) * ((gy[..., None] - 0.45) / 0.55))
    fill = np.broadcast_to(fill, (ch, cw, 3)).copy()
    # inner shading: darken near the bottom edge of each glyph, highlight near top
    inner_dn = g * (1 - (shift(g, 0, -5 * ss)))
    inner_up = g * (1 - (shift(g, 0, 5 * ss)))
    fill = fill * (1 - 0.35 * blur(inner_dn, 2 * ss)[..., None]) + 0.5 * blur(inner_up, 2 * ss)[..., None]

    ext_col = np.broadcast_to(hexc("#5a1a8c"), (ch, cw, 3)) * (0.55 + 0.6 * gy[..., None])
    outl = hexc(cfg["outline"])

    rgb = np.zeros((ch, cw, 3), np.float32)
    a = np.zeros((ch, cw), np.float32)
    # drop shadow + warm glow
    sh = blur(shift(o, 6 * ss, 10 * ss), 10 * ss)
    rgb = over(rgb, np.zeros(3), sh * 0.7)
    a = np.maximum(a, sh * 0.7)
    glow = blur(o, 22 * ss)
    rgb = screen(rgb, hexc("#ff7a2a")[None, None] * glow[..., None] * 0.6)
    a = np.maximum(a, glow * 0.6)
    rgb = over(rgb, np.broadcast_to(outl, rgb.shape), o)
    rgb = over(rgb, ext_col, e)
    rgb = over(rgb, fill, g)
    a = np.maximum(a, o)

    im = Image.fromarray((np.dstack([np.clip(rgb, 0, 1), a]) * 255).astype(np.uint8), "RGBA")
    im = im.rotate(cfg["rotate"], resample=Image.BICUBIC, expand=True)
    full = Image.new("RGBA", (W, H))
    x, y = cfg["xy"][0] * ss - pad + 0, cfg["xy"][1] * ss - pad
    full.alpha_composite(im, (int(x), int(y))) if x >= 0 and y >= 0 else full.paste(im, (int(x), int(y)), im)
    arr = np.asarray(full, np.float32) / 255
    return arr


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(os.path.join(OUT, "layers"), exist_ok=True)
    W0, H0 = CONFIG["size"]
    ss = CONFIG["ss"]
    W, H = W0 * ss, H0 * ss
    rng = np.random.default_rng(11)
    saved = []

    def save(name, arr):
        p = os.path.join(OUT, "layers", name + ".png")
        mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
        Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), mode).save(p)
        saved.append((name, p))

    print("rendering 3-D models...")
    cave = build_cave(CONFIG["cave"], ss, W, H)

    print("compositing...")
    cc = CONFIG["cave"]
    img = cave["rgba"][..., :3].copy()
    lmask = cave["tags"].get("lonecard", 0) + cave["tags"].get("pickaxe", 0)
    omask = cave["tags"].get("ore", np.zeros((H, W), np.float32))
    # Lonecard: coloured rim light (cyan from the ore side, violet on the back)
    img = img + rim(lmask, 1, -0.4, 4 * ss, hexc(cc["rim_right"]), 1.1) \
              + rim(lmask, -1, -0.5, 4 * ss, hexc(cc["rim_left"]), 1.0)
    # bloom: ore gems + lava, plus a soft halo behind the ore stack (focal point)
    img = screen(img, hexc("#2ff0ff")[None, None] * (blur(omask, 46 * ss) * 0.85 * (1 - 0.8 * omask))[..., None])
    bloom = blur(cave["emis"], 4 * ss) * 0.8 + blur(cave["emis"], 30 * ss) * 0.8
    img = screen(np.clip(img, 0, 1), np.clip(bloom, 0, 1))
    # light haze in the air: cyan around the ore, orange over the lava chamber
    dg = CONFIG["diamond_glow"]
    gx, gy = (c * ss for c in dg["center"])
    img = screen(img, hexc(dg["color"])[None, None] * (radial(W, H, gx, gy, dg["radius"] * ss, 1.6) * 0.3 * (1 - 0.75 * omask))[..., None])
    lx, ly = (c * ss for c in CONFIG["lava_glow"]["center"])
    img = screen(img, hexc("#ff6a1a")[None, None] * (radial(W, H, lx, ly, CONFIG["lava_glow"]["radius"] * ss, 1.5) * 0.38)[..., None])

    # --- Raven (background): thin lava-coloured rim so he separates, but stays secondary
    rc = CONFIG["raven"]
    ra = cave["tags"].get("raven", np.zeros((H, W), np.float32))
    img = img + rim(ra, 1, -0.6, 3 * ss, hexc(rc["rim_right"]), 1.3) \
              + rim(ra, -0.8, -0.6, 2 * ss, hexc(rc["rim_left"]), 0.7)
    img = np.clip(img, 0, 1)

    # --- depth of field: soften everything far behind the hero
    depth = cave["depth"]
    far = np.clip((depth - 95) / 120, 0, 1) * (1 - 0.5 * ra)
    img = img * (1 - far[..., None]) + blur(img, 2.2 * ss) * far[..., None]
    save("01_scene", img)

    # --- particles
    sp = sparkles(W, H, ss, [
        (880, 430, 150, 30, "#bffcff", 2.0, 5.5),    # diamond sparkle around the ore
        (700, 560, 90, 14, "#5ff4ff", 1.5, 4),
        (930, 420, 240, 26, "#ffb35a", 1.2, 3.5),    # embers rising off the lava
    ], rng)
    save("04_particles", np.dstack([sp, sp.max(-1)]))
    img = screen(img, sp)

    # --- vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vig = 1 - 0.55 * (((xx / W - 0.5) * 1.6) ** 2 + ((yy / H - 0.5) * 1.7) ** 2)
    img = img * np.clip(vig, 0.35, 1)[..., None]

    # --- title
    if CONFIG["text"].get("enabled"):
        txt = text_layer(W, H, ss, CONFIG["text"])
        save("05_text", txt)
        img = over(img, txt[..., :3], txt[..., 3])

    # --- grade: slight contrast + saturation pop
    lum = img.mean(-1, keepdims=True)
    img = np.clip(lum + (img - lum) * 1.12, 0, 1)
    img = np.clip((img - 0.5) * 1.06 + 0.5, 0, 1)

    final = Image.fromarray((img * 255 + 0.5).astype(np.uint8)).resize((W0, H0), Image.LANCZOS)
    final_path = os.path.join(OUT, "thumbnail.png")
    final.save(final_path, optimize=True)
    final.resize((400, 225), Image.LANCZOS).save(os.path.join(OUT, "thumbnail_400.png"))

    # --- layered OpenRaster file (Krita / GIMP / MyPaint)
    ora = os.path.join(OUT, "thumbnail_layers.ora")
    with zipfile.ZipFile(ora, "w") as z:
        z.writestr("mimetype", "image/openraster", compress_type=zipfile.ZIP_STORED)
        stack = "".join(
            f'<layer name="{n}" src="data/{n}.png" x="0" y="0" '
            f'{"composite-op=\"svg:screen\" " if n == "04_particles" else ""}/>'
            for n, _ in reversed(saved))
        z.writestr("stack.xml", f'<?xml version="1.0"?><image w="{W}" h="{H}"><stack>{stack}</stack></image>')
        for n, p in saved:
            z.write(p, f"data/{n}.png")
        buf = io.BytesIO(); final.save(buf, "PNG"); z.writestr("mergedimage.png", buf.getvalue())
        buf = io.BytesIO(); final.resize((256, 144)).save(buf, "PNG"); z.writestr("Thumbnails/thumbnail.png", buf.getvalue())

    kb = os.path.getsize(final_path) / 1024
    print(f"saved {final_path}  {final.size[0]}x{final.size[1]}  {kb:.0f} KB")


if __name__ == "__main__":
    main()
