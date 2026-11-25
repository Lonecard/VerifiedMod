"""
mcrender.py - a small software renderer for Minecraft-style cube models.

* Real 3D boxes with the standard 64x64 skin UV layout (classic + slim arms,
  second/overlay layer inflated by 0.5px, alpha cut-out).
* Perspective-correct texture mapping with NEAREST ("Closest") sampling, so
  skin pixels stay perfectly sharp and are never stretched.
* Flat per-face lighting: ambient + key (diffuse) + coloured rim lights
  (additive sheen, so it reads even on black skins) + point lights.
* Returns RGBA + per-pixel tag ids, so the compositor can build masks
  (glow, rim, etc.) for each object separately.

Units: 1 unit = 1 skin pixel. A full block is 16 units, a player is 32 tall.
World axes: +x = viewer's right, +y = up, +z = towards the viewer.
"""
import math
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# math helpers
# --------------------------------------------------------------------------- #
def rot(rx=0.0, ry=0.0, rz=0.0):
    """Rotation matrix from Euler angles in degrees. Applied as Ry @ Rx @ Rz."""
    ax, ay, az = (math.radians(a) for a in (rx, ry, rz))
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def normalize(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


class Xform:
    """Affine transform p' = R @ (s * p) + t."""

    def __init__(self, R=None, t=(0, 0, 0), s=1.0):
        self.R = np.eye(3) if R is None else np.asarray(R, float)
        self.t = np.asarray(t, float)
        self.s = float(s)

    def __matmul__(self, other):  # self applied after other
        return Xform(self.R @ other.R, self.R @ (self.s * other.t) + self.t, self.s * other.s)

    def apply(self, p):
        return (np.asarray(p, float) * self.s) @ self.R.T + self.t

    def apply_dir(self, d):
        return np.asarray(d, float) @ self.R.T


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
class Face:
    __slots__ = ("pts", "tex", "rect", "normal", "tag", "emissive", "shade")

    def __init__(self, pts, tex, rect, normal, tag, emissive=0.0, shade=True):
        self.pts, self.tex, self.rect, self.normal = pts, tex, rect, normal
        self.tag, self.emissive, self.shade = tag, emissive, shade


def box_faces(x0, y0, z0, w, h, d, u, v, tex, xf, tag, inflate=0.0,
              emissive=0.0, skip=()):
    """
    Six faces of a Minecraft box with the standard skin unwrap at (u, v):

        [    ][top ][bot ][    ]       row v      height d
        [rght][frnt][left][back]       row v+d    height h

    Each face lists its corners in texture order TL, TR, BR, BL so the skin
    maps exactly as in-game (character faces +z; its right side is -x).
    """
    X0, Y0, Z0 = x0 - inflate, y0 - inflate, z0 - inflate
    X1, Y1, Z1 = x0 + w + inflate, y0 + h + inflate, z0 + d + inflate
    P = lambda *c: np.array(c, float)
    faces = {
        "front": ([P(X0, Y1, Z1), P(X1, Y1, Z1), P(X1, Y0, Z1), P(X0, Y0, Z1)], (u + d, v + d, w, h), (0, 0, 1)),
        "back": ([P(X1, Y1, Z0), P(X0, Y1, Z0), P(X0, Y0, Z0), P(X1, Y0, Z0)], (u + 2 * d + w, v + d, w, h), (0, 0, -1)),
        "right": ([P(X0, Y1, Z0), P(X0, Y1, Z1), P(X0, Y0, Z1), P(X0, Y0, Z0)], (u, v + d, d, h), (-1, 0, 0)),
        "left": ([P(X1, Y1, Z1), P(X1, Y1, Z0), P(X1, Y0, Z0), P(X1, Y0, Z1)], (u + d + w, v + d, d, h), (1, 0, 0)),
        "top": ([P(X0, Y1, Z0), P(X1, Y1, Z0), P(X1, Y1, Z1), P(X0, Y1, Z1)], (u + d, v, w, d), (0, 1, 0)),
        "bottom": ([P(X0, Y0, Z1), P(X1, Y0, Z1), P(X1, Y0, Z0), P(X0, Y0, Z0)], (u + d + w, v, w, d), (0, -1, 0)),
    }
    out = []
    for name, (pts, rect, n) in faces.items():
        if name in skip:
            continue
        out.append(Face(xf.apply(np.array(pts)), tex, rect, normalize(xf.apply_dir(n)),
                        tag, emissive))
    return out


def load_tex(path):
    return np.asarray(Image.open(path).convert("RGBA"), np.float32) / 255.0


def is_slim(skin):
    """Slim (Alex) arms leave the 4th column of the right-arm front empty."""
    return skin[20:32, 54:56, 3].max() < 0.5 and skin[20:32, 50, 3].max() > 0.5


# --------------------------------------------------------------------------- #
# player model
# --------------------------------------------------------------------------- #
DEFAULT_POSE = dict(head=(0, 0, 0), body=(0, 0, 0), arm_r=(0, 0, 0), arm_l=(0, 0, 0),
                    leg_r=(0, 0, 0), leg_l=(0, 0, 0))


def player_faces(skin, root, pose=None, tag="player", slim=None, overlay=True):
    """
    Build a posed player. `root` is an Xform placing the model (origin between
    the feet). `pose` maps part -> (rx, ry, rz) degrees around the joint.
    Returns (faces, joints) - joints are Xforms so items can be attached.
    """
    p = dict(DEFAULT_POSE, **(pose or {}))
    slim = is_slim(skin) if slim is None else slim
    aw = 3 if slim else 4
    faces, joints = [], {}

    def part(name, pivot, box, uv, uv2, parent=root):
        xf = parent @ Xform(rot(*p[name]), pivot)
        joints[name] = xf
        bx, by, bz, w, h, d = box
        faces.extend(box_faces(bx, by, bz, w, h, d, *uv, skin, xf, tag))
        if overlay and uv2 is not None:
            faces.extend(box_faces(bx, by, bz, w, h, d, *uv2, skin, xf, tag, inflate=0.25 if name != "head" else 0.5))
        return xf

    body = part("body", (0, 12, 0), (-4, 0, -2, 8, 12, 4), (16, 16), (16, 32))
    part("head", (0, 12, 0), (-4, 0, -4, 8, 8, 8), (0, 0), (32, 0), parent=body)
    part("arm_r", (-4 - aw + aw / 2 + 0.0, 10, 0), (-aw / 2 - 0.0, -12 + 2, -2, aw, 12, 4), (40, 16), (40, 32), parent=body)
    part("arm_l", (4 + aw / 2, 10, 0), (-aw / 2, -10, -2, aw, 12, 4), (32, 48), (48, 48), parent=body)
    part("leg_r", (-2, 12, 0), (-2, -12, -2, 4, 12, 4), (0, 16), (0, 32))
    part("leg_l", (2, 12, 0), (-2, -12, -2, 4, 12, 4), (16, 48), (0, 48))
    return faces, joints


def sprite_voxels(sprite, xf, tag, depth=1.0, emissive=0.0):
    """Extrude a 2-D RGBA sprite (H,W,4 float) into voxels, like held items.
    Sprite x -> +x, sprite y (down) -> -y, extruded along z."""
    H, W = sprite.shape[:2]
    a = sprite[..., 3] > 0.5
    faces = []
    for y in range(H):
        for x in range(W):
            if not a[y, x]:
                continue
            skip = []
            if x > 0 and a[y, x - 1]: skip.append("right")
            if x < W - 1 and a[y, x + 1]: skip.append("left")
            if y > 0 and a[y - 1, x]: skip.append("top")
            if y < H - 1 and a[y + 1, x]: skip.append("bottom")
            # every face of the voxel samples the same sprite texel
            for f in box_faces(x, H - 1 - y, -depth / 2, 1, 1, depth, 0, 0, sprite, xf, tag,
                               emissive=emissive, skip=skip):
                f.rect = (x, y, 1, 1)
                faces.append(f)
    return faces


def block_faces(tex16, x, y, z, xf, tag, size=16, emissive=0.0, skip=()):
    """A full cube using the same 16x16 texture on every side."""
    out = []
    for f in box_faces(x, y, z, size, size, size, 0, 0, tex16, xf, tag, emissive=emissive, skip=tuple(skip)):
        f.rect = (0, 0, tex16.shape[1], tex16.shape[0])
        out.append(f)
    return out


# --------------------------------------------------------------------------- #
# camera + lights + rasteriser
# --------------------------------------------------------------------------- #
class Camera:
    def __init__(self, target, yaw=0, pitch=0, dist=100, fov=30):
        """Orbit camera: yaw/pitch in degrees around `target`."""
        self.target = np.asarray(target, float)
        R = rot(-pitch, yaw, 0)              # camera-to-world rotation
        self.pos = self.target + R @ np.array([0, 0, dist])
        self.R = R.T                          # world-to-camera
        self.fov = fov

    @classmethod
    def look(cls, pos, target, fov=50, up=(0, 1, 0)):
        """Camera at `pos` looking at `target`."""
        c = cls.__new__(cls)
        c.pos, c.target, c.fov = np.asarray(pos, float), np.asarray(target, float), fov
        fwd = normalize(c.target - c.pos)
        right = normalize(np.cross(fwd, up))
        upv = np.cross(right, fwd)
        c.R = np.stack([right, upv, -fwd])
        return c


class Lights:
    def __init__(self, ambient=(0.35, 0.35, 0.4), key=None, rims=(), points=(), sheen=None, fog=None):
        self.fog = fog        # (rgb, density per unit of depth) or None
        self.ambient = np.asarray(ambient, float)
        self.key = key        # (dir towards light, rgb)
        self.rims = rims      # list of (dir towards light, rgb, power)
        self.points = points  # list of (pos, rgb, radius)
        self.sheen = sheen    # additive key: (dir, rgb) - lets black materials show planes


def shade_face(f, lights, center):
    """Per-face terms (ambient, key, sheen, rims). Point lights are per pixel."""
    if not f.shade:
        return np.ones(3), np.zeros(3)
    n = f.normal
    mult = lights.ambient.copy()
    add = np.zeros(3)
    if lights.key is not None:
        d, c = lights.key
        mult += np.asarray(c) * max(0.0, float(n @ normalize(d)))
    if lights.sheen is not None:
        d, c = lights.sheen
        add += np.asarray(c) * max(0.0, float(n @ normalize(d)))
    for d, c, pw in lights.rims:
        k = max(0.0, float(n @ normalize(d)))
        add += np.asarray(c) * (k ** pw)
    return mult, add


def point_terms(n, wp, lights):
    """Per-pixel point lights for world positions wp (N,3) on a face with normal n."""
    mult = np.zeros((len(wp), 3), np.float32)
    add = np.zeros((len(wp), 3), np.float32)
    for pos, c, rad in lights.points:
        L = np.asarray(pos, np.float32) - wp
        dist = np.sqrt((L * L).sum(1)) + 1e-6
        k = np.maximum(0.0, (L @ n) / dist) / (1 + (dist / rad) ** 2)
        c = np.asarray(c, np.float32)
        mult += k[:, None] * c
        add += k[:, None] * c * 0.2
    return mult, add


def render(faces, cam, W, H, lights, ss=1):
    """Rasterise faces. Returns dict(rgba float HxWx4, tag int HxW, emis HxWx3,
    depth HxW, tags list). Pixel-centre sampling; nearest-neighbour texels.
    Textures may carry a 5th channel = emission mask (per-texel glow)."""
    W, H = W * ss, H * ss
    rgb = np.zeros((H, W, 3), np.float32)
    emis = np.zeros((H, W, 3), np.float32)
    zbuf = np.zeros((H, W), np.float32)          # stores 1/depth, 0 = empty
    tagbuf = np.full((H, W), -1, np.int16)
    tags = []
    f_px = 0.5 * H / math.tan(math.radians(cam.fov) / 2)
    cx, cy = W / 2, H / 2
    fog = getattr(lights, "fog", None)

    for face in faces:
        center = face.pts.mean(0)
        if float(face.normal @ (cam.pos - center)) <= 0:      # back-face cull
            continue
        pc = (face.pts - cam.pos) @ cam.R.T
        z = -pc[:, 2]
        if (z < 1e-3).any():
            continue
        sx = cx + f_px * pc[:, 0] / z
        sy = cy - f_px * pc[:, 1] / z
        mult, add = shade_face(face, lights, center)
        if face.tag not in tags:
            tags.append(face.tag)
        tid = tags.index(face.tag)
        u0, v0, tw, th = face.rect
        tex = face.tex
        st = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        for tri in ((0, 1, 2), (0, 2, 3)):
            X, Y, Z = sx[list(tri)], sy[list(tri)], z[list(tri)]
            xmin, xmax = int(max(0, math.floor(X.min()))), int(min(W - 1, math.ceil(X.max())))
            ymin, ymax = int(max(0, math.floor(Y.min()))), int(min(H - 1, math.ceil(Y.max())))
            if xmin > xmax or ymin > ymax:
                continue
            area = (X[1] - X[0]) * (Y[2] - Y[0]) - (X[2] - X[0]) * (Y[1] - Y[0])
            if abs(area) < 1e-9:
                continue
            gx, gy = np.meshgrid(np.arange(xmin, xmax + 1) + 0.5, np.arange(ymin, ymax + 1) + 0.5)
            b0 = ((X[1] - gx) * (Y[2] - gy) - (X[2] - gx) * (Y[1] - gy)) / area
            b1 = ((X[2] - gx) * (Y[0] - gy) - (X[0] - gx) * (Y[2] - gy)) / area
            b2 = 1 - b0 - b1
            eps = -1e-4
            inside = (b0 >= eps) & (b1 >= eps) & (b2 >= eps)
            if not inside.any():
                continue
            iz = b0 / Z[0] + b1 / Z[1] + b2 / Z[2]
            S = st[list(tri)]
            s = (b0 * S[0, 0] / Z[0] + b1 * S[1, 0] / Z[1] + b2 * S[2, 0] / Z[2]) / iz
            t = (b0 * S[0, 1] / Z[0] + b1 * S[1, 1] / Z[1] + b2 * S[2, 1] / Z[2]) / iz
            tx = np.clip(np.floor(s * tw), 0, tw - 1).astype(int) + int(u0)
            ty = np.clip(np.floor(t * th), 0, th - 1).astype(int) + int(v0)
            texel = tex[ty, tx]
            sub_z = zbuf[ymin:ymax + 1, xmin:xmax + 1]
            m = inside & (texel[..., 3] > 0.5) & (iz > sub_z)
            if not m.any():
                continue
            tc = texel[m]                                   # (N, C)
            col = tc[:, :3] * mult + add
            if face.tag in AO_TAGS:                         # soft darkening toward block edges
                e = np.minimum(np.minimum(s[m], 1 - s[m]), np.minimum(t[m], 1 - t[m]))
                col = col * (1 - 0.32 * np.clip(1 - e / 0.18, 0, 1) ** 2)[:, None]
            if lights.points and face.shade:
                P = face.pts[list(tri)]
                bb = np.stack([b0[m] / Z[0], b1[m] / Z[1], b2[m] / Z[2]], 1) / iz[m][:, None]
                wp = (bb @ P).astype(np.float32)
                pm, pa = point_terms(face.normal.astype(np.float32), wp, lights)
                col = col + tc[:, :3] * pm + pa
            em = np.zeros(len(tc), np.float32)
            if face.emissive:
                em = tc[:, 4] if tc.shape[1] > 4 else np.ones(len(tc), np.float32)
                glow = tc[:, :3] * (0.6 + 0.5 * mult.mean() + face.emissive)
                col = col * (1 - em[:, None]) + glow * em[:, None]
            if fog is not None:
                fcol, dens = fog
                fz = 1 - np.exp(-(1 / iz[m]) * dens)
                fz = fz * (1 - em)                          # glowing texels cut through fog
                col = col * (1 - fz[:, None]) + np.asarray(fcol, np.float32) * fz[:, None]
            rgb[ymin:ymax + 1, xmin:xmax + 1][m] = col
            emis[ymin:ymax + 1, xmin:xmax + 1][m] = tc[:, :3] * (em * face.emissive)[:, None]
            sub_z[m] = iz[m]
            tagbuf[ymin:ymax + 1, xmin:xmax + 1][m] = tid

    alpha = (zbuf > 0).astype(np.float32)
    rgba = np.dstack([np.clip(rgb, 0, 1), alpha])
    depth = np.where(zbuf > 0, 1 / np.maximum(zbuf, 1e-9), np.inf).astype(np.float32)
    return dict(rgba=rgba, tag=tagbuf, tags=tags, emis=emis, depth=depth)


def voxel_faces(cells, xf, tag_of=lambda t: "world", emissive_of=lambda t: 0.0, textures=None):
    """cells: {(i,j,k): texture-key}. Builds 16-unit blocks, emitting only the
    faces that touch an empty cell (no hidden interior faces)."""
    out = []
    nb = {"right": (-1, 0, 0), "left": (1, 0, 0), "bottom": (0, -1, 0), "top": (0, 1, 0),
          "back": (0, 0, -1), "front": (0, 0, 1)}
    for (i, j, k), key in cells.items():
        skip = [n for n, (a, b, c) in nb.items() if (i + a, j + b, k + c) in cells]
        if len(skip) == 6:
            continue
        tex = textures[key] if textures else key
        out += block_faces(tex, i * 16, j * 16, k * 16, xf, tag_of(key),
                           emissive=emissive_of(key), skip=skip)
    return out


AO_TAGS = {"world", "ore"}


def to_image(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))
