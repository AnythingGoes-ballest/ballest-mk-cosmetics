"""Builds the MK Characters from scratch: Mario, Yoshi, Peach and Bowser as balls, each a .glb of simple shapes made
here (no game or downloaded models), its model text file, a tile picture and the plugin icon.

    python tools/make_models.py            # everything
    python tools/make_models.py mario      # one character (and its pictures)

Each character is worn as a ball: the game's own clear ball is hidden (`ball hidden`) and the character is the whole
ball, in a `travel` group: it stays upright and faces where the ball goes (on the Customize page it faces the camera).
Its body is a sphere of the ball's size (radius 50), so it matches the ball the game rolls.
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meshkit import (BALL, Mesh, Patch, cone, d, edge, ellipsoid, extrude, frame, lathe, lip, render, rot, sheet,
                     tube, unit, axes_z, write_glb)

ROOT = Path(__file__).resolve().parent.parent
MODELS, PREVIEWS = ROOT / "models", ROOT / "previews"


def on(e, a, r=BALL):
    return d(e, a) * r


def eye(m, e, a, w, h, iris, irisw, irish, pupil="pupil", white="white", r=BALL, look=(0, 0), depth=2.4,
        shine=True):
    """A cartoon eye on the surface: a white oval, a coloured iris, a pupil and a highlight."""
    n = d(e, a)
    ax = frame(n)
    c = n * r
    ellipsoid(m, white, c, (w, h, depth), ax, nu=8, nv=16)
    ic = c + ax[0] * look[0] + ax[1] * look[1] + n * (depth - 0.9)
    ellipsoid(m, iris, ic, (irisw, irish, 1.2), ax, nu=6, nv=14)
    pc = ic + n * 0.55
    ellipsoid(m, pupil, pc, (irisw * 0.48, irish * 0.52, 0.9), ax, nu=6, nv=12)
    if shine:
        hc = ic + ax[0] * (-irisw * 0.25) + ax[1] * (irish * 0.38) + n * 1.0
        ellipsoid(m, "white", hc, (irisw * 0.28, irish * 0.22, 0.6), ax, nu=4, nv=8)


def surface_path(points, r):
    """Points (elevation, azimuth) joined smoothly on a sphere of radius r (or one radius per point)."""
    pts = np.asarray(points, float)
    rs = np.broadcast_to(np.asarray(r, float), (len(pts),))
    t = np.linspace(0, len(pts) - 1, (len(pts) - 1) * 4 + 1)
    e = np.interp(t, np.arange(len(pts)), pts[:, 0])
    a = np.interp(t, np.arange(len(pts)), pts[:, 1])
    rr = np.interp(t, np.arange(len(pts)), rs)
    return d(e, a) * rr[:, None]


def cap_edge(patch, az):
    """The elevation where a patch's edge crosses each azimuth (coming up from below)."""
    es = np.linspace(-90, 90, 3601)
    out = []
    for a in np.atleast_1d(az):
        dirs = d(es, np.full_like(es, a))
        th = np.degrees(np.arccos(np.clip(dirs @ patch.A, -1, 1)))
        phi = np.arctan2(dirs @ patch.C, dirs @ patch.B)
        inside = th <= patch.alpha(phi)
        out.append(es[np.argmax(inside)])
    return np.array(out)


def band(s, s0=0.0, width=0.12, tuck=-0.6):
    """1 inside an annulus or disc patch, falling below the surface at its edges."""
    E = edge(1 - s, width)
    if s0 > 0:
        E = E * edge(s - s0, width)
    return E, tuck * (1 - E)


# ------------------------------------------------------------------------------------------------------- Mario
def mario():
    m = Mesh({
        "skin": {"colour": "#fcc496", "rough": 0.6},
        "nose": {"colour": "#f7b583", "rough": 0.5},
        "cap": {"colour": "#e01c1c", "rough": 0.55},
        "white": {"colour": "#fbfbfb", "rough": 0.45},
        "hair": {"colour": "#4b2410", "rough": 0.8},
        "tache": {"colour": "#1d120c", "rough": 0.75},
        "iris": {"colour": "#2a6fe0", "rough": 0.2},
        "pupil": {"colour": "#08080c", "rough": 0.2},
        "shirt": {"colour": "#d81a1a", "rough": 0.6},
        "overalls": {"colour": "#2848c8", "rough": 0.6},
        "button": {"colour": "#ffd02a", "rough": 0.3, "metal": 0.0},
    })
    ellipsoid(m, "skin", (0, 0, 0), (BALL, BALL, BALL), nu=16, nv=28)

    # the cap: a puffy dome sitting back on the head, its front higher than its back
    def cap_thick(s, phi):
        f, t = lip(s, 0.93, -1.0)
        puff = 4.2 + 3.6 * (1 + np.cos(phi)) / 2 * (1 - s ** 2)
        return puff * f + t
    cap = Patch(d(67.5, 180), 72.5, cap_thick, front=(1, 0, 0))
    cap.build(m, "cap", ns=12, nphi=36)
    # the brim, flat and wide over the eyes, tipped down a little
    pitch = math.radians(18)
    fwd = np.array([math.cos(pitch), 0, -math.sin(pitch)])
    upv = np.array([math.sin(pitch), 0, math.cos(pitch)])
    ellipsoid(m, "cap", (36, 0, 30), (27, 19, 2.6), np.array([[0, 1.0, 0], fwd, upv]), nu=8, nv=28)
    # the emblem: a white circle on the front of the cap with a red M
    E = d(58, 0)
    emb = Patch(E, 17, lambda s, phi: 0.45 * lip(s, 0.85, -1.6)[0] + lip(s, 0.85, -1.6)[1], base=cap.radius_at)
    emb.build(m, "white", ns=8, nphi=36)
    M = np.array([(-1, -1), (-0.6, -1), (-0.6, 0.38), (0, -0.32), (0.6, 0.38), (0.6, -1), (1, -1), (1, 1),
                  (0.62, 1), (0, 0.3), (-0.62, 1), (-1, 1)]) * np.array([8.4, 7.4])
    extrude(m, "cap", M, E * (cap.radius_at(E) + 0.45), frame(E), -1.2, 0.7)

    # eyes under the brim, blue, close together
    for s in (-1, 1):
        eye(m, 12, 12.5 * s, 6.2, 9.6, "iris", 4.3, 6.6, look=(0.6 * s, -0.4))
    # thick dark eyebrows just under the brim
    for s in (-1, 1):
        path = surface_path([(24.5, 5 * s), (26.8, 12 * s), (25.2, 19.5 * s)], 51.9)
        tube(m, "tache", path, np.array([1.3] + [2.0] * (len(path) - 2) + [1.3]), nv=8, flat=(d(28, 12 * s), 0.55))
    # the nose: big and round
    ellipsoid(m, "nose", on(-2, 0, 49), (13.5, 11.5, 12.5), frame(d(-2, 0)), nu=12, nv=20)
    # the moustache: bushy lobes under the nose, out toward the cheeks
    for a, e, w, h in ((5, -15.5, 9.5, 8.5), (17, -15, 9.5, 8.5), (29, -12.5, 9, 8), (39, -9, 7.5, 7)):
        for s in (-1, 1):
            n = d(e, a * s)
            ax = frame(n)
            ax = np.array([ax[0], ax[1], ax[2]])
            ellipsoid(m, "tache", n * 51.5, (w, h, 4.6), ax, nu=6, nv=10)
    # ears, and sideburns in front of them
    for s in (-1, 1):
        n = d(6, 83 * s)
        ellipsoid(m, "skin", n * 48.5, (5.5, 9.5, 5), frame(n), nu=8, nv=14)
        n = d(10, 66 * s)
        ellipsoid(m, "hair", n * 49.6, (5.0, 11.0, 2.8), frame(n), nu=6, nv=12)

    # hair at the back, under the cap, in tufts
    az = np.linspace(92, 268, 46)
    ec = cap_edge(cap, az)
    u = (az - 92) / 176
    v = np.linspace(0, 1, 8)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    top = ec[:, None] + 6
    bottom = ec[:, None] - (12 + 7 * np.abs(np.sin(math.pi * 6.5 * U)))
    elev = top + (bottom - top) * Vv
    A = np.repeat(az[:, None], len(v), 1)
    Ed = edge(U, 0.06) * edge(1 - U, 0.06) * edge(1 - Vv, 0.35)
    sheet(m, "hair", d(elev, A), BALL + 3.6 * Ed - 0.6 * (1 - Ed))

    # his red shirt collar and blue overalls with yellow buttons, round the bottom of the head
    shirt = Patch((0, 0, -1), 36, lambda s, phi: 2.2 * band(s, 0.45)[0] + band(s, 0.45)[1], s0=0.45)
    shirt.build(m, "shirt", ns=8, nphi=40)
    for a in (0, 180):
        bib = Patch(d(-66, a), 10, lambda s, phi: 3.4 * band(s, 0, 0.2)[0] + band(s, 0, 0.2)[1])
        bib.build(m, "overalls", ns=6, nphi=24)
    for s in (-1, 1):
        n = d(-61, 8 * s)
        ellipsoid(m, "button", n * 53.2, (2.4, 2.4, 1.3), frame(n), nu=5, nv=10)
    return m


# ------------------------------------------------------------------------------------------------------- Peach
def peach():
    m = Mesh({
        "skin": {"colour": "#ffd6ba", "rough": 0.55},
        "hair": {"colour": "#f4bc24", "rough": 0.45},
        "gold": {"colour": "#f6c41e", "rough": 0.3},
        "ruby": {"colour": "#e0102c", "rough": 0.1},
        "sapphire": {"colour": "#1f52e8", "rough": 0.1},
        "white": {"colour": "#fbfbfb", "rough": 0.45},
        "iris": {"colour": "#3b8cf0", "rough": 0.2},
        "pupil": {"colour": "#10204a", "rough": 0.2},
        "lash": {"colour": "#1a1210", "rough": 0.6},
        "brow": {"colour": "#c9953a", "rough": 0.6},
        "lips": {"colour": "#f0607e", "rough": 0.35},
        "blush": {"colour": "#ffc4bc", "rough": 0.6},
        "dress": {"colour": "#ff8cc4", "rough": 0.5},
        "trim": {"colour": "#f25aa6", "rough": 0.5},
    })
    ellipsoid(m, "skin", (0, 0, 0), (BALL, BALL, BALL), nu=16, nv=28)

    # hair: a full blonde mop over the top, sides and back, framing the face; long at the back
    az = np.linspace(-180, 180, 49)
    fold = np.abs(az)
    bottom = np.interp(fold, [0, 15, 30, 42, 55, 68, 90, 140, 180], [41, 38, 29, 15, -6, -28, -48, -60, -62])
    v = np.linspace(0, 1, 12)
    A, Vv = np.meshgrid(az, v, indexing="ij")
    elev = 90 + (bottom[:, None] - 90) * Vv
    puff = np.interp(np.abs(A), [0, 40, 70, 180], [3.0, 4.5, 7.0, 7.5])
    E = edge(1 - Vv, 0.18)
    sheet(m, "hair", d(elev, A), BALL + puff * (0.8 + 0.2 * np.cos(Vv * math.pi * 0.5)) * E - 0.6 * (1 - E))
    # locks of hair down her back, so she reads from behind too
    for la in (128, 146, 163, 180, 197, 214, 232):
        lb = float(np.interp(abs(la - 360 if la > 180 else la), [0, 15, 30, 42, 55, 68, 90, 140, 180],
                             [41, 38, 29, 15, -6, -28, -48, -60, -62]))
        el = np.linspace(48, lb + 3, 10)
        vv = (90 - el) / (90 - lb)
        pf = float(np.interp(abs(((la + 180) % 360) - 180), [0, 40, 70, 180], [3.0, 4.5, 7.0, 7.5]))
        rr = BALL + pf * (0.8 + 0.2 * np.cos(vv * math.pi * 0.5)) * edge(1 - vv, 0.18) + 0.4
        path = d(el, np.full_like(el, la)) * rr[:, None]
        tube(m, "hair", path, np.r_[np.linspace(1.5, 3.4, 4), np.full(4, 3.4), [2.2, 0.6]], nv=8)
    # bangs swept to each side from a parting in the middle
    for s in (-1, 1):
        path = surface_path([(66, 2 * s), (50, 12 * s), (38, 24 * s), (25, 36 * s), (10, 45 * s)],
                            [53.0, 53.2, 53.0, 52.6, 52.0])
        tube(m, "hair", path, np.linspace(7.0, 3.0, len(path)), nv=10, flat=(d(40, 24 * s), 0.42))
    # curls flipping out at the sides, low beside the face
    for s in (-1, 1):
        path = surface_path([(-6, 68 * s), (-26, 78 * s), (-40, 86 * s), (-44, 98 * s), (-38, 106 * s)],
                            [55, 56, 57, 59, 60])
        tube(m, "hair", path, np.array(list(np.linspace(6.5, 5, len(path) - 4)) + [4.5, 3.8, 2.8, 0.5]), nv=12)

    # the crown, on top and tipped a little forward: a gold band with points, a ruby at the front and sapphires
    n = d(70, 0)
    base = n * 55.0
    ax = axes_z(n, ref=(1, 0, 0))
    k = 1.3
    lathe(m, "gold", np.array([(10.5, -2), (12.2, -2), (13.6, 6), (12.4, 6.6), (11.0, 5.8), (9.6, -1)]) * k, base, ax,
          nv=30, closed=True, smooth=False)
    for i in range(5):
        phi = math.radians(i * 72)
        out = math.cos(phi) * ax[0] + math.sin(phi) * ax[1]
        side = np.cross(ax[2], out)
        pc = base + ax[2] * 5.5 * k + out * 12.9 * k
        tri = np.array([(-4.6, 0), (4.6, 0), (0, 8.5)]) * k
        extrude(m, "gold", tri, pc, np.array([side, ax[2], out]), -0.8, 0.8)
        ellipsoid(m, "gold", pc + ax[2] * 9.4 * k, (2.0, 2.0, 2.0), nu=5, nv=8)
        gem = base + ax[2] * 2.2 * k + out * 13.6 * k
        if i == 0:
            ellipsoid(m, "ruby", gem, (3.6, 4.4, 2.0), np.array([side, ax[2], out]), nu=6, nv=12)
        else:
            ellipsoid(m, "sapphire", gem, (2.5, 2.5, 1.6), np.array([side, ax[2], out]), nu=5, nv=10)
    ellipsoid(m, "ruby", base + ax[2] * 4.0 * k, (8.5 * k, 8.5 * k, 6.0 * k), ax, nu=8, nv=14)   # the velvet inside

    # big blue eyes with lashes, thin brows, a little nose, pink lips, rosy cheeks
    for s in (-1, 1):
        eye(m, 11, 15 * s, 8.0, 11.0, "iris", 6.2, 8.6, look=(0.4 * s, -0.6))
        lash = surface_path([(17.5, 7.5 * s), (22.5, 13 * s), (22.0, 19 * s), (19.5, 23.5 * s), (22, 26.5 * s)], 52.0)
        tube(m, "lash", lash, np.array([0.7] + [1.25] * (len(lash) - 2) + [0.4]), nv=6, flat=(d(22, 15 * s), 0.5))
        brow = surface_path([(29, 9 * s), (31.5, 15 * s), (30.5, 22 * s)], 50.6)
        tube(m, "brow", brow, np.array([0.5] + [0.95] * (len(brow) - 2) + [0.4]), nv=6, flat=(d(31, 15 * s), 0.5))
        nb = d(-6, 27 * s)
        ellipsoid(m, "blush", nb * 49.6, (5.5, 3.6, 1.0), frame(nb), nu=5, nv=10)
    ellipsoid(m, "skin", on(-2, 0, 49.5), (3.4, 4.2, 3.6), frame(d(-2, 0)), nu=6, nv=10)
    nl = d(-16, 0)
    ellipsoid(m, "lips", nl * 49.6, (5.0, 2.2, 1.6), frame(nl), nu=5, nv=12)

    # blue earrings hanging below the hair
    for s in (-1, 1):
        ne = d(-30, 72 * s)
        ellipsoid(m, "gold", ne * 51.5, (1.3, 1.3, 1.3), nu=4, nv=8)
        ellipsoid(m, "sapphire", d(-36, 73 * s) * 52.5, (3.2, 3.2, 3.2), nu=7, nv=12)

    # her pink dress round the bottom: a darker pink trim and a blue brooch at the front
    dress = Patch((0, 0, -1), 38, lambda s, phi: 2.4 * band(s, 0.42)[0] + band(s, 0.42)[1], s0=0.42)
    dress.build(m, "dress", ns=8, nphi=40)
    trim = Patch((0, 0, -1), 39, lambda s, phi: 3.0 * band(s, 0.86, 0.06)[0] + band(s, 0.86, 0.06)[1], s0=0.86)
    trim.build(m, "trim", ns=4, nphi=40)
    nb = d(-57, 0)
    ellipsoid(m, "gold", nb * 51.8, (6.2, 5.0, 1.8), frame(nb), nu=6, nv=14)
    ellipsoid(m, "sapphire", nb * 53.2, (4.8, 3.8, 1.6), frame(nb), nu=6, nv=14)
    return m


# ------------------------------------------------------------------------------------------------------- Yoshi
def yoshi():
    m = Mesh({
        "green": {"colour": "#46b92c", "rough": 0.5},
        "white": {"colour": "#fbfbfb", "rough": 0.45},
        "saddle": {"colour": "#e8261c", "rough": 0.35},
        "spike": {"colour": "#f27a14", "rough": 0.45},
        "shoe": {"colour": "#f2661a", "rough": 0.35},
        "pupil": {"colour": "#0c0c0c", "rough": 0.2},
        "nostril": {"colour": "#1e5a12", "rough": 0.6},
    })
    # his round body is the ball: a white belly in front, a red saddle with a white rim on his back
    ellipsoid(m, "green", (0, 0, 0), (BALL, BALL, BALL), nu=16, nv=28)
    belly = Patch(d(-18, 0), 46, lambda s, phi: 1.6 * lip(s, 0.85, -0.8)[0] + lip(s, 0.85, -0.8)[1])
    belly.build(m, "white", ns=8, nphi=30)
    sad = Patch(d(42, 180), 34, lambda s, phi: (2.5 + 4.5 * (1 - s ** 2)) * lip(s, 0.9, -0.8)[0] + lip(s, 0.9, -0.8)[1])
    sad.build(m, "saddle", ns=12, nphi=40)
    # the saddle's white rim: a roll round its edge
    phi = np.linspace(-math.pi, math.pi, 49)
    rim = sad.dirs(np.full_like(phi, 0.97), phi) * (BALL + 1.2)
    tube(m, "white", rim, 2.6, nv=8)
    # orange spikes down his back below the saddle
    for e, h in ((3, 11.0), (-13, 10.5), (-28, 9.0), (-41, 7.0)):
        n = d(e, 180)
        cone(m, "spike", n * 48.5, unit(n + np.array([0, 0, -0.3])), 6.0, h, nv=10)

    # the head, up on the front of the body: a big round snout, a white chin, eyes on top
    H = np.array([16.0, 0, 54.0])
    ellipsoid(m, "green", H, (25, 24, 25), nu=14, nv=24)
    ellipsoid(m, "green", (44, 0, 60), (22, 21, 18), nu=14, nv=24)               # the snout
    for s in (-1, 1):
        ellipsoid(m, "nostril", (62.5, 6.5 * s, 66), (1.8, 2.4, 3.0), nu=5, nv=8)
    ellipsoid(m, "saddle", (40, 0, 44), (13, 15, 3.2), nu=6, nv=16)              # his mouth
    ellipsoid(m, "white", (35, 0, 47.5), (21, 21.5, 7.5), nu=8, nv=18)          # cheeks and chin
    for s in (-1, 1):
        ec = np.array([24.0, 9.5 * s, 82.0])
        tilt = rot((0, 1, 0), -12).T
        ellipsoid(m, "white", ec, (8.5, 8.0, 12.5), tilt, nu=10, nv=18)
        ellipsoid(m, "pupil", ec + np.array([7.4, 0.8 * s, 0.5]), (2.2, 3.6, 6.6), tilt, nu=6, nv=12)
        ellipsoid(m, "white", ec + np.array([8.6, 0.6 * s, 3.2]), (0.8, 1.2, 1.6), nu=4, nv=8)
    # spikes on the back of his head
    for x, z, up in ((0.0, 74.0, 0.6), (-7.0, 63.0, 0.15)):
        cone(m, "spike", np.array([x, 0, z]), unit(np.array([-1.0, 0, up])), 5.8, 10.0, nv=10)

    # red-orange shoes peeking out in front, and a little tail behind
    for s in (-1, 1):
        ellipsoid(m, "shoe", (30, 22 * s, -38), (16, 10.5, 9), rot((0, 0, 1), 12 * s).T, nu=8, nv=16)
    tube(m, "green", np.array([(-46.0, 0, -22), (-56, 0, -28), (-64, 0, -30)]), np.array([10, 6, 0.5]), nv=12)
    return m


# ------------------------------------------------------------------------------------------------------ Bowser
def bowser():
    m = Mesh({
        "skin": {"colour": "#f0b134", "rough": 0.55},
        "belly": {"colour": "#f3e1a6", "rough": 0.6},
        "groove": {"colour": "#c99a4e", "rough": 0.7},
        "shell": {"colour": "#2f8f2c", "rough": 0.4},
        "rim": {"colour": "#f2e4b8", "rough": 0.55},
        "spike": {"colour": "#fafaf2", "rough": 0.3},
        "horn": {"colour": "#f2e4b8", "rough": 0.45},
        "hair": {"colour": "#ec3c16", "rough": 0.55},
        "white": {"colour": "#fbfbfb", "rough": 0.4},
        "iris": {"colour": "#d01818", "rough": 0.2},
        "pupil": {"colour": "#0a0a0a", "rough": 0.2},
        "mouth": {"colour": "#7a1010", "rough": 0.6},
        "cuff": {"colour": "#232326", "rough": 0.35},
        "nostril": {"colour": "#6a3a10", "rough": 0.6},
    })
    ellipsoid(m, "skin", (0, 0, 0), (BALL, BALL, BALL), nu=16, nv=28)
    # his spiked green shell on his back, with a cream rim; spikes on cream rings
    sh = Patch(d(14, 180), 74, lambda s, phi: (2.5 + 9.5 * (1 - s ** 2)) * lip(s, 0.9, -0.8)[0] + lip(s, 0.9, -0.8)[1])
    sh.build(m, "shell", ns=12, nphi=40)
    phi = np.linspace(-math.pi, math.pi, 41)
    rim = sh.dirs(np.full_like(phi, 0.975), phi) * (BALL + 1.6)
    tube(m, "rim", rim, 3.8, nv=8)
    spots = [(0.0, 0.0)] + [(0.5, k * 60.0) for k in range(6)] + [(0.82, k * 40.0 + 20) for k in range(9)]
    for s_, ph in spots:
        n = sh.dirs(np.array(s_), np.radians(np.array(ph)))
        if n[2] < -0.42:
            continue                                    # too low: it would dig into the ground
        r = float(sh.radius(np.array(s_), np.radians(np.array(ph))))
        lathe(m, "rim", [(7.0, -1.5), (7.6, 0.6), (6.4, 1.8), (4.6, 1.2), (4.4, -1.5)], n * (r - 0.4), axes_z(n),
              nv=10, closed=True)
        cone(m, "spike", n * (r + 0.6), n, 4.8, 11.5, nv=10)
    # his cream belly with grooves across it
    belly = Patch(d(-8, 0), 48, lambda s, phi: 2.0 * lip(s, 0.86, -0.8)[0] + lip(s, 0.86, -0.8)[1])
    belly.build(m, "belly", ns=10, nphi=36)
    for e in (-30, -14, 2, 18):
        half = max(10.0, min(40.0, 0.82 * math.sqrt(max(0, 46 ** 2 - (e + 8) ** 2))))
        path = surface_path([(e, -half), (e - 1.5, 0), (e, half)], 52.0)
        tube(m, "groove", path, 0.8, nv=5)

    # the head, up on the front: a wide snout with a cream jaw, fangs, red eyes, angry brows, horns and a red mane
    H = np.array([18.0, 0, 56.0])
    ellipsoid(m, "skin", H, (24, 25, 23), nu=12, nv=20)
    ellipsoid(m, "skin", (38, 0, 57), (18, 20, 12.5), nu=10, nv=18)                # snout
    ellipsoid(m, "belly", (36, 0, 46), (16, 18, 7.5), nu=8, nv=20)                 # jaw
    ellipsoid(m, "mouth", (42, 0, 50.5), (10, 16, 3.0), nu=6, nv=16)
    for s in (-1, 1):
        cone(m, "white", np.array([49.0, 8.5 * s, 52.0]), (0.15, 0, -1), 2.2, 6.0, nv=8)    # fangs
        cone(m, "white", np.array([47.0, 13.0 * s, 47.0]), (0.15, 0, 1), 1.8, 4.5, nv=8)
        ellipsoid(m, "nostril", (54.5, 6.0 * s, 62), (1.4, 2.4, 1.6), nu=4, nv=8)
        # eyes
        ec = np.array([33.5, 10.5 * s, 67.0])
        n = unit(np.array([0.85, 0.25 * s, 0.35]))
        ax = frame(n)
        ellipsoid(m, "white", ec, (4.6, 6.2, 3.0), ax, nu=7, nv=14)
        ic = ec + n * 2.0 + ax[1] * -0.6
        ellipsoid(m, "iris", ic, (3.2, 4.4, 1.4), ax, nu=6, nv=12)
        ellipsoid(m, "pupil", ic + n * 0.7, (1.4, 2.6, 0.9), ax, nu=5, nv=10)
        # brows: thick red, slanting down to the middle
        brow = np.array([(37.5, 4.0 * s, 71.0), (36.0, 10.0 * s, 74.5), (32.5, 16.0 * s, 77.5)])
        tube(m, "hair", brow, np.array([2.2, 2.8, 1.6]), nv=8)
        # horns: cream, out to the sides and curving up
        horn = np.array([(16.0, 18.0 * s, 70.0), (14.0, 27.0 * s, 77.0), (11.0, 32.0 * s, 87.0), (6.0, 32.0 * s, 96.0)])
        t = np.linspace(0, 1, 9)
        path = np.array([np.interp(t, np.linspace(0, 1, 4), horn[:, c]) for c in range(3)]).T
        tube(m, "horn", path, np.linspace(6.5, 0.2, len(path)), nv=10)
    # the mane: red spikes standing up and back over the head
    for p, dirn, r, h in (((10, 0, 76), (0.25, 0, 1), 6.0, 15), ((4, -9, 72), (-0.4, -0.6, 1), 5.0, 12),
                          ((4, 9, 72), (-0.4, 0.6, 1), 5.0, 12), ((0, 0, 70), (-0.8, 0, 0.8), 6.0, 15),
                          ((-4, -8, 62), (-1, -0.5, 0.4), 5.0, 12), ((-4, 8, 62), (-1, 0.5, 0.4), 5.0, 12),
                          ((-6, 0, 56), (-1, 0, 0.1), 5.0, 12)):
        cone(m, "hair", np.array(p, float), unit(dirn), r, h, nv=10)

    # arms with black spiked cuffs and claws, and feet with claws
    for s in (-1, 1):
        arm = np.array([(4.0, 40.0 * s, 16.0), (12.0, 50.0 * s, -2.0), (20.0, 50.0 * s, -16.0)])
        tube(m, "skin", arm, np.array([10.0, 9.5, 9.0]), nv=12)
        ellipsoid(m, "skin", (22, 50 * s, -20), (9.5, 9, 8.5), nu=8, nv=14)
        wrist = np.array([16.0, 50.5 * s, -9.0])
        wn = unit(np.array([0.5, 0.0, -1.0]))
        wa = axes_z(wn)
        lathe(m, "cuff", [(10.2, -2.6), (11.4, -1.6), (11.4, 1.6), (10.2, 2.6), (9.0, 0)], wrist, wa, nv=20,
              closed=True)
        for k in range(5):
            ang = math.radians(k * 72 + 18)
            out = math.cos(ang) * wa[0] + math.sin(ang) * wa[1]
            cone(m, "spike", wrist + out * 10.8, out, 2.6, 6.5, nv=8)
        for s2 in (-1, 0, 1):
            cone(m, "spike", np.array([29.0, 50 * s + 4.5 * s2, -23.0]), (1, 0.1 * s2, -0.3), 1.6, 4.5, nv=6)
        ellipsoid(m, "skin", (30, 24 * s, -40), (15, 11, 8), nu=8, nv=16)
        for s2 in (-1, 0, 1):
            cone(m, "spike", np.array([43.0, 24 * s + 5 * s2, -42.0]), (1, 0.15 * s2, -0.15), 1.8, 5.0, nv=6)
    return m


CHARACTERS = {
    "mario": ("Mario", mario, "Mario: red cap with his M, moustache, big nose, blue eyes, brown hair, red shirt and blue overalls."),
    "yoshi": ("Yoshi", yoshi, "Yoshi: green with a white belly and chin, big snout, eyes on top, red saddle, orange spikes and shoes."),
    "peach": ("Peach", peach, "Peach: gold crown with a ruby and sapphires, blonde hair, blue eyes, sapphire earrings, pink dress and brooch."),
    "bowser": ("Bowser", bowser, "Bowser: spiked green shell with a cream rim, horns, red mane and brows, red eyes, fangs, spiked cuffs."),
}


def write(key):
    name, build, desc = CHARACTERS[key]
    m = build()
    write_glb(m, MODELS / f"{key}.glb")
    lo, hi = m.bounds()
    c = (lo + hi) / 2
    text = (f"# {desc}\n"
            f"# Made from scratch by tools/make_models.py ({m.triangles()} triangles). The game's clear ball is hidden: the\n"
            f"# character is the whole ball, its head or body a sphere of the ball's size, kept upright and facing where\n"
            f"# the ball goes.\n"
            f"ball hidden\n"
            f"group {key} travel\n"
            f"mesh {key}.glb at={c[0]:.3f},{c[1]:.3f},{lo[2]:.3f}\n")
    (MODELS / f"{key}.txt").write_text(text, newline="\n")
    pic = render(m, size=256, azimuth=-28, elevation=10, extent=max(66, np.max(np.abs(np.r_[lo, hi])) * 1.02),
                 centre=(0, 0, (lo[2] + hi[2]) / 2))
    pic.save(PREVIEWS / f"{key}.png")
    print(f"{key}: {m.triangles()} triangles, bounds {np.round(lo, 1)} .. {np.round(hi, 1)}")
    return pic


def main():
    keys = sys.argv[1:] or list(CHARACTERS)
    MODELS.mkdir(exist_ok=True)
    PREVIEWS.mkdir(exist_ok=True)
    pics = {k: write(k) for k in keys}
    if len(pics) == len(CHARACTERS):
        icon = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        for i, k in enumerate(CHARACTERS):
            icon.alpha_composite(pics[k], ((i % 2) * 256, (i // 2) * 256))
        icon.save(ROOT / "icon.png")


if __name__ == "__main__":
    main()
