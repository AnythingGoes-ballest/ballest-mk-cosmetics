"""A small mesh kit for building the characters from scratch: shapes made in numpy, written as glTF 2.0 (.glb) for the
plugin manager's model reader, and drawn as tile pictures.

Coordinates are the game's: cm, the ball's middle at the origin, its radius 50, +x forward, +y the ball's right, +z up.
The glTF file is written in glTF's axes (metres, +Y up, +Z forward), which the host turns back with
(x, y, z) -> (100z, -100x, 100y) (meshfile.cpp ToGame).
"""
import json
import math
import struct

import numpy as np
from PIL import Image

BALL = 50.0


# ----------------------------------------------------------------------------------------------------- vectors
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def d(e, a):
    """The direction at elevation e and azimuth a (degrees): azimuth 0 forward, 90 right; elevation 90 up."""
    e, a = np.radians(e), np.radians(a)
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1)


def frame(n, up=(0, 0, 1)):
    """Axes (rows: right, up, normal) for a part lying on a surface whose outward normal is n: right and up as someone
    looking at the surface sees them (the game's axes are left-handed, so facing -x, +y is on the left)."""
    n = unit(n)
    up = np.asarray(up, float)
    if abs(np.dot(n, unit(up))) > 0.999:
        up = np.array([-1.0, 0, 0]) if n[2] > 0 else np.array([1.0, 0, 0])
    r = unit(np.cross(n, up))
    u = np.cross(r, n)
    return np.array([r, u, n])


def rot(axis, deg):
    """Rotation matrix about an axis (right-hand rule in these coordinates)."""
    a = unit(axis)
    t = math.radians(deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * (k @ k)


def axes_z(n, ref=(1, 0, 0)):
    """Axes (rows x, y, z) whose z is n: for shapes built along z (lathes, cones)."""
    z = unit(n)
    ref = np.asarray(ref, float)
    if abs(np.dot(z, unit(ref))) > 0.99:
        ref = np.array([0, 0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0, 0])
    x = unit(ref - np.dot(ref, z) * z)
    y = np.cross(z, x)
    return np.array([x, y, z])


# ------------------------------------------------------------------------------------------------------ meshes
class Mesh:
    """Triangles by material: each material is one glTF primitive."""

    def __init__(self, materials):
        self.materials = materials          # name -> dict(colour="#rrggbb", rough=, metal=)
        self.parts = {}                     # name -> list of (V, F, smooth)

    def add(self, mat, V, F, smooth=True):
        assert mat in self.materials, mat
        V = np.asarray(V, float)
        F = np.asarray(F, np.int64)
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        keep = np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-7
        self.parts.setdefault(mat, []).append((V, F[keep], smooth))

    def triangles(self):
        return sum(len(F) for parts in self.parts.values() for _, F, _ in parts)

    def bounds(self):
        V = np.concatenate([self.merged(mat)[0] for mat in self.parts])     # every point the file holds
        return V.min(0), V.max(0)

    def merged(self, mat):
        """One material's triangles: positions, normals, indices (smooth parts welded by position)."""
        Ps, Ns, Is, base = [], [], [], 0
        for V, F, smooth in self.parts.get(mat, []):
            if smooth:
                key = np.round(V, 4)
                _, inv = np.unique(key, axis=0, return_inverse=True)
                inv = inv.reshape(-1)
                fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])     # area weighted
                acc = np.zeros((inv.max() + 1, 3))
                for k in range(3):
                    np.add.at(acc, inv[F[:, k]], fn)
                N = unit(acc[inv] + 1e-12)
                P, I = V, F
            else:
                P = V[F.reshape(-1)]
                fn = unit(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]))
                N = np.repeat(fn, 3, axis=0)
                I = np.arange(len(P)).reshape(-1, 3)
            Ps.append(P)
            Ns.append(N)
            Is.append(I + base)
            base += len(P)
        return np.concatenate(Ps), np.concatenate(Ns), np.concatenate(Is)


def grid(P, wrap_u=False, wrap_v=False):
    """Quads between the rows and columns of a (nu, nv, 3) grid of points, as triangles."""
    nu, nv = P.shape[:2]
    idx = np.arange(nu * nv).reshape(nu, nv)
    F = []
    for i in range(nu if wrap_u else nu - 1):
        i2 = (i + 1) % nu
        for j in range(nv if wrap_v else nv - 1):
            j2 = (j + 1) % nv
            a, b, c, e = idx[i, j], idx[i2, j], idx[i2, j2], idx[i, j2]
            F.append((a, b, c))
            F.append((a, c, e))
    return P.reshape(-1, 3), np.array(F)


def orient(V, F, inside=None):
    """Turns the triangles to face outward: away from `inside` (a point) when given, else by the signed volume."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    if inside is None:
        flip = np.sum(np.einsum("ij,ij->i", a, n)) < 0
    else:
        flip = np.sum(np.einsum("ij,ij->i", (a + b + c) / 3 - inside, n)) < 0
    return F[:, ::-1].copy() if flip else F


def cap_ends(V, F, rings, centres):
    """Closes a tube or lathe's open ends with a fan to a middle point."""
    V = list(V)
    F = list(map(tuple, F))
    for ring, ctr in zip(rings, centres):
        k = len(V)
        V.append(ctr)
        for j in range(len(ring)):
            F.append((k, ring[j], ring[(j + 1) % len(ring)]))
    return np.array(V), np.array(F)


def ellipsoid(m, mat, c, radii, axes=None, nu=10, nv=16):
    axes = np.eye(3) if axes is None else np.asarray(axes, float)
    th = np.linspace(0, math.pi, nu + 1)
    ph = np.linspace(0, 2 * math.pi, nv, endpoint=False)
    T, Ph = np.meshgrid(th, ph, indexing="ij")
    local = np.stack([np.sin(T) * np.cos(Ph), np.sin(T) * np.sin(Ph), np.cos(T)], -1) * np.asarray(radii, float)
    P = np.asarray(c, float) + local @ axes
    V, F = grid(P, wrap_v=True)
    m.add(mat, V, orient(V, F, inside=np.asarray(c, float)))


def lathe(m, mat, profile, c, axes, nv=20, closed=False, smooth=True):
    """A shape turned about local z from a profile of (radius, height) points; `closed` joins the last to the first."""
    pr = np.asarray(profile, float)
    ph = np.linspace(0, 2 * math.pi, nv, endpoint=False)
    R, Ph = np.meshgrid(pr[:, 0], ph, indexing="ij")
    Z = np.repeat(pr[:, 1:2], nv, 1)
    local = np.stack([R * np.cos(Ph), R * np.sin(Ph), Z], -1)
    P = np.asarray(c, float) + local @ np.asarray(axes, float)
    V, F = grid(P, wrap_u=closed, wrap_v=True)
    mid = np.asarray(c, float) + np.array([0, 0, pr[:, 1].mean()]) @ np.asarray(axes, float)
    if closed:
        F = orient(V - mid, F)
    else:
        rings, centres = [], []
        for i in (0, len(pr) - 1):
            if pr[i, 0] > 1e-6:
                rings.append(list(range(i * nv, (i + 1) * nv)))
                centres.append(np.asarray(c, float) + np.array([0, 0, pr[i, 1]]) @ np.asarray(axes, float))
        V, F = cap_ends(V, F, rings, centres)
        F = orient(V - mid, F)
    m.add(mat, V, F, smooth)


def cone(m, mat, base, n, r, h, top=0.0, nv=12):
    lathe(m, mat, [(r, 0), (top, h)] if top > 0 else [(r, 0), (r * 0.02, h * 0.995), (0, h)], base, axes_z(n), nv)


def tube(m, mat, path, radii, nv=12, flat=None):
    """A tube along a path of points with a radius at each (0 makes a point); `flat` squashes it across (0..1) toward
    the given direction, for ribbons such as eyebrows and lashes."""
    path = np.asarray(path, float)
    radii = np.broadcast_to(np.asarray(radii, float), (len(path),))
    T = np.gradient(path, axis=0)
    T = unit(T)
    n0 = np.cross(T[0], [0, 0, 1.0])
    if np.linalg.norm(n0) < 1e-3:
        n0 = np.cross(T[0], [1.0, 0, 0])
    N = [unit(n0)]
    for i in range(1, len(path)):           # parallel transport
        v = N[-1] - np.dot(N[-1], T[i]) * T[i]
        N.append(unit(v))
    N = np.array(N)
    B = np.cross(T, N)
    ph = np.linspace(0, 2 * math.pi, nv, endpoint=False)
    rings = []
    for i in range(len(path)):
        nn, bb = N[i], B[i]
        if flat is not None:
            f = unit(np.asarray(flat[0], float) - np.dot(flat[0], T[i]) * T[i])
            g = np.cross(T[i], f)
            nn, bb = f * flat[1], g
        rings.append(path[i] + radii[i] * (np.cos(ph)[:, None] * nn + np.sin(ph)[:, None] * bb))
    P = np.array(rings)
    V, F = grid(P, wrap_v=True)
    ends, centres = [], []
    if radii[0] > 1e-6:
        ends.append(list(range(nv))[::-1])
        centres.append(path[0])
    if radii[-1] > 1e-6:
        ends.append(list(range((len(path) - 1) * nv, len(path) * nv)))
        centres.append(path[-1])
    V, F = cap_ends(V, F, ends, centres)
    # each ring faces away from its own path point
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    cen = (a + b + c) / 3
    near = np.argmin(np.linalg.norm(cen[:, None, :] - path[None, :, :], axis=2), axis=1)
    s = np.einsum("ij,ij->i", cen - path[near], n)
    if np.sum(s) < 0:
        F = F[:, ::-1].copy()
    m.add(mat, V, F)


def extrude(m, mat, poly, origin, axes, z0, z1):
    """A flat polygon (2D points, counter-clockwise) pushed out from z0 to z1 along the local z of `axes`."""
    poly = np.asarray(poly, float)
    tris = ear_clip(poly)
    n = len(poly)
    V = np.concatenate([np.c_[poly, np.full(n, z0)], np.c_[poly, np.full(n, z1)]])
    F = [(t[2], t[1], t[0]) for t in tris] + [(t[0] + n, t[1] + n, t[2] + n) for t in tris]
    for i in range(n):
        j = (i + 1) % n
        F += [(i, j, j + n), (i, j + n, i + n)]
    V = np.asarray(origin, float) + V @ np.asarray(axes, float)
    F = np.array(F)
    m.add(mat, V, fix_closed(V, F), smooth=False)


def fix_closed(V, F):
    c = V.mean(0)
    vol = np.sum(np.einsum("ij,ij->i", V[F[:, 0]] - c, np.cross(V[F[:, 1]] - c, V[F[:, 2]] - c)))
    return F[:, ::-1].copy() if vol < 0 else F


def ear_clip(poly):
    idx = list(range(len(poly)))
    tris = []

    def cross2(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross2(a, b, c) <= 1e-9:
                continue
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = poly[j]
                if cross2(a, b, p) >= 0 and cross2(b, c, p) >= 0 and cross2(c, a, p) >= 0:
                    inside = True
                    break
            if not inside:
                tris.append((i0, i1, i2))
                idx.pop(k)
                break
    tris.append(tuple(idx))
    return tris


class Patch:
    """A piece lying on a round surface around an axis: everything within an angle of the axis (the angle may change
    around it), raised by a thickness that tucks under at the edge. `s` runs from the middle (0) to the edge (1), `phi`
    around it from the side `front` points to."""

    def __init__(self, axis, alpha, thick, base=lambda dirs: np.full(dirs.shape[:-1], BALL), front=(1, 0, 0),
                 s0=0.0):
        self.A = unit(axis)
        f = np.asarray(front, float)
        if abs(np.dot(unit(f), self.A)) > 0.99:
            f = np.array([0, 0, 1.0])
        self.B = unit(f - np.dot(f, self.A) * self.A)
        self.C = np.cross(self.A, self.B)
        self.alpha = alpha if callable(alpha) else (lambda phi, a=alpha: np.full_like(phi, a, dtype=float))
        self.thick = thick
        self.base = base
        self.s0 = s0

    def dirs(self, s, phi):
        th = np.radians(s * self.alpha(phi))
        return (np.cos(th)[..., None] * self.A
                + np.sin(th)[..., None] * (np.cos(phi)[..., None] * self.B + np.sin(phi)[..., None] * self.C))

    def radius(self, s, phi):
        return self.base(self.dirs(s, phi)) + self.thick(s, phi)

    def radius_at(self, dirs):
        """The outer radius at directions inside the patch."""
        dirs = unit(dirs)
        th = np.degrees(np.arccos(np.clip(dirs @ self.A, -1, 1)))
        phi = np.arctan2(dirs @ self.C, dirs @ self.B)
        s = th / self.alpha(phi)
        return self.base(dirs) + self.thick(np.clip(s, 0, 1), phi)

    def build(self, m, mat, ns=14, nphi=40):
        s = np.linspace(self.s0, 1, ns + 1)
        phi = np.linspace(-math.pi, math.pi, nphi, endpoint=False)
        S, Ph = np.meshgrid(s, phi, indexing="ij")
        P = self.dirs(S, Ph) * self.radius(S, Ph)[..., None]
        V, F = grid(P, wrap_v=True)
        m.add(mat, V, orient(V, F, inside=np.zeros(3)))


def sheet(m, mat, dirs, radii):
    """A sheet over a (nu, nv) grid of directions at the given radii (from the ball's middle), facing outward."""
    P = dirs * radii[..., None]
    V, F = grid(P)
    m.add(mat, V, orient(V, F, inside=np.zeros(3)))


def edge(x, width):
    """0 at the edge (x = 0) rising smoothly to 1 at x = width."""
    t = np.clip(np.asarray(x, float) / width, 0, 1)
    return t * t * (3 - 2 * t)


def lip(s, start=0.92, tuck=-0.6):
    """A thickness factor: 1 across the patch, then down past the surface at the edge (s = 1), so no gap shows."""
    t = np.clip((s - start) / (1 - start), 0, 1)
    return 1 - t * t * (3 - 2 * t), tuck * t


# --------------------------------------------------------------------------------------------------------- glTF
def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


def to_gltf(P):
    """Game cm (x fwd, y right, z up) -> glTF metres (x, y up, z fwd): the inverse of the host's ToGame."""
    return np.stack([-P[:, 1], P[:, 2], P[:, 0]], 1) / 100.0


def write_glb(m, path):
    bin_ = bytearray()
    views, accessors, prims, materials = [], [], [], []

    def view(data, target):
        while len(bin_) % 4:
            bin_.append(0)
        views.append({"buffer": 0, "byteOffset": len(bin_), "byteLength": len(data), "target": target})
        bin_.extend(data)
        return len(views) - 1

    for name, spec in m.materials.items():
        if name not in m.parts:
            continue
        P, N, I = m.merged(name)
        Pg = to_gltf(P).astype(np.float32)
        Ng = (to_gltf(N) * 100).astype(np.float32)
        # The axis change is a mirror (determinant -1), so the winding is reversed to stay counter-clockwise outward.
        Ig = I[:, ::-1].astype(np.uint32)
        vol = np.sum(np.einsum("ij,ij->i", Pg[Ig[:, 0]], np.cross(Pg[Ig[:, 1]], Pg[Ig[:, 2]])))
        pv = view(Pg.tobytes(), 34962)
        accessors.append({"bufferView": pv, "componentType": 5126, "count": len(Pg), "type": "VEC3",
                          "min": Pg.min(0).tolist(), "max": Pg.max(0).tolist()})
        pa = len(accessors) - 1
        nvw = view(Ng.tobytes(), 34962)
        accessors.append({"bufferView": nvw, "componentType": 5126, "count": len(Ng), "type": "VEC3"})
        na = len(accessors) - 1
        iv = view(Ig.reshape(-1).tobytes(), 34963)
        accessors.append({"bufferView": iv, "componentType": 5125, "count": int(Ig.size), "type": "SCALAR"})
        ia = len(accessors) - 1
        lin = srgb_to_linear(hex_rgb(spec["colour"]))
        mat = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": [*map(float, lin), 1.0],
                                                      "metallicFactor": float(spec.get("metal", 0.0)),
                                                      "roughnessFactor": float(spec.get("rough", 0.5))}}
        if spec.get("glow"):
            mat["emissiveFactor"] = [*map(float, lin)]
        materials.append(mat)
        prims.append({"attributes": {"POSITION": pa, "NORMAL": na}, "indices": ia, "material": len(materials) - 1})
        spec["_volume"] = float(vol)
    gltf = {"asset": {"version": "2.0", "generator": "ballest-mk-cosmetics make_models.py"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "character"}],
            "meshes": [{"primitives": prims}], "materials": materials, "accessors": accessors,
            "bufferViews": views, "buffers": [{"byteLength": len(bin_)}]}
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    while len(bin_) % 4:
        bin_.append(0)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(bin_), 0x004E4942) + bytes(bin_)
    with open(path, "wb") as f:
        f.write(out)


# ------------------------------------------------------------------------------------------------------ pictures
def render(m, size=256, ss=4, azimuth=-25, elevation=12, extent=None, centre=None, background=None):
    """The model drawn smooth-shaded in its colours (orthographic, from in front and a little to one side), on a
    transparent square; `azimuth` is where the viewer stands (0 = in front, on +x, looking back at the model)."""
    W = size * ss
    view = -d(elevation, azimuth)                      # looking toward the model
    # The game's axes are left-handed (x forward, y right, z up): a viewer facing -x has +y on their left.
    right = unit(np.cross([0, 0, 1.0], view))
    up = np.cross(view, right)
    lo, hi = m.bounds()
    ctr = (lo + hi) / 2 if centre is None else np.asarray(centre, float)
    ext = (np.max(hi - lo) / 2 * 1.08) if extent is None else extent
    zbuf = np.full((W, W), np.inf)
    img = np.zeros((W, W, 4))
    light = unit(np.array([0.6, -0.3, 0.9]) - 0.6 * view)
    for name, spec in m.materials.items():
        if name not in m.parts:
            continue
        P, N, I = m.merged(name)
        col = hex_rgb(spec["colour"])
        q = P - ctr
        sx = (q @ right / ext * 0.5 + 0.5) * W
        sy = (0.5 - q @ up / ext * 0.5) * W
        sz = q @ view
        diff = np.clip(N @ light, 0, 1)
        spec_k = 0.35 * (1 - float(spec.get("rough", 0.5))) ** 2
        half = unit(light - view)
        shine = np.clip(N @ half, 0, 1) ** 40 * spec_k
        shade = 0.42 + 0.62 * diff
        vcol = np.clip(col[None, :] * shade[:, None] + shine[:, None], 0, 1)
        if spec.get("glow"):
            vcol = np.repeat(col[None, :], len(P), 0)
        for t in I:
            xs, ys, zs = sx[t], sy[t], sz[t]
            x0, x1 = int(max(0, np.floor(xs.min()))), int(min(W - 1, np.ceil(xs.max())))
            y0, y1 = int(max(0, np.floor(ys.min()))), int(min(W - 1, np.ceil(ys.max())))
            if x0 > x1 or y0 > y1:
                continue
            X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
            if abs(den) < 1e-9:
                continue
            w0 = ((ys[1] - ys[2]) * (X - xs[2]) + (xs[2] - xs[1]) * (Y - ys[2])) / den
            w1 = ((ys[2] - ys[0]) * (X - xs[2]) + (xs[0] - xs[2]) * (Y - ys[2])) / den
            w2 = 1 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            z = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
            sub = zbuf[y0:y1 + 1, x0:x1 + 1]
            win = inside & (z < sub)
            if not win.any():
                continue
            sub[win] = z[win]
            c = w0[..., None] * vcol[t[0]] + w1[..., None] * vcol[t[1]] + w2[..., None] * vcol[t[2]]
            tile = img[y0:y1 + 1, x0:x1 + 1]
            tile[win, :3] = c[win]
            tile[win, 3] = 1
    out = (img.reshape(size, ss, size, ss, 4).mean((1, 3)))
    rgb = np.where(out[..., 3:] > 0, out[..., :3] / np.maximum(out[..., 3:], 1e-6), 0)
    pic = np.concatenate([rgb, out[..., 3:]], -1)
    im = Image.fromarray((np.clip(pic, 0, 1) * 255).astype(np.uint8), "RGBA")
    if background is not None:
        bg = Image.new("RGBA", im.size, background)
        bg.alpha_composite(im)
        im = bg
    return im
