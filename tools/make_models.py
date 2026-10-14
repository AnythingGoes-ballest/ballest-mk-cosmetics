"""Builds the MK Characters from scratch: Mario, Yoshi, Peach and Bowser as balls, each made of moving parts (a .glb of
simple shapes made here for each part: no game or downloaded models), its model text file with the rig and the
moves, a tile picture and the plugin icon.

    python tools/make_models.py                # everything
    python tools/make_models.py mario          # one character (and its pictures)
    python tools/make_models.py --sheets       # also a picture of every move, in previews/moves/ (not shipped)

Each character is worn as a ball: the game's own clear ball is hidden (`ball hidden`) and the character is the whole
ball, in a `travel` group: it stays upright and faces where the ball goes (on the Customize page it faces the camera).
Its body (or head) is a sphere of the ball's size (radius 50), so at rest it matches the ball the game rolls.

The rig, the same for all four:
    <name> travel                   upright, facing where the ball goes
      lean   pivot on the ground    leans into the run as the ball speeds up, bounces each step; hops and spins
        body pivot in the middle    twists with the stride; turns to look around
          head, arms, feet, tail... swing with the stride (arms against the legs), and the moves' keys
The tempo makes the stride quicker and bigger as the ball speeds up, and a small calm sway at rest. The moves are
short animations the host picks at random for what the ball is doing: rolling, still, in the air, and at once on a
jump or a landing (host 0.27.0's `move` and `key`).
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meshkit import (BALL, Mesh, Patch, cone, d, edge, ellipsoid, extrude, frame, lathe, lip, render, rot, sheet,
                     tube, unit, axes_z, write_glb_nodes)
from rig import Rig

ROOT = Path(__file__).resolve().parent.parent
MODELS, PREVIEWS = ROOT / "models", ROOT / "previews"

# Strides a second: 0.6 at rest (a slow sway), 0.25 more for every m/s, at most 3.2; swings a fifth of their size at
# rest, full from 8 m/s.
TEMPO = dict(rate=0.6, run=0.25, max=3.2, calm=0.2, full=8)


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


def ring_y(m, mat, c, s, r0, r1, half, nv=16):
    """A cuff: a short closed ring around an axis along y (toward the body), centred on c."""
    ax = axes_z(np.array([0, -s, 0.0]))
    lathe(m, mat, [(r0, -half), (r1, -half * 0.6), (r1, half * 0.6), (r0, half), (r0 * 0.9, 0)], c, ax, nv=nv,
          closed=True)


def glove(m, mat, c, s, big=1.0):
    """A white cartoon glove, a fist with a thumb up front and a cuff toward the body (s: -1 left, 1 right)."""
    c = np.asarray(c, float)
    ellipsoid(m, mat, c, np.array([7.5, 6.5, 7.0]) * big, nu=8, nv=14)
    ellipsoid(m, mat, c + np.array([5.5, -2.5 * s, 3.0]) * big, np.array([3.4, 2.6, 2.6]) * big, nu=5, nv=10)
    ring_y(m, mat, c + np.array([0, -6.5 * s, 1.0]) * big, s, 5.6 * big, 6.8 * big, 2.4 * big)


def skeleton(R, lean=10, bob=3.0, twist=7):
    """The rig's spine: travelling, leaning and bouncing from the ground, twisting about the middle. The body's mesh."""
    R.group(R.key, travel=True, mesh=False)
    R.group("lean", on=R.key, pivot=(0, 0, -BALL), mesh=False, lean=lean, bob=bob)
    return R.group("body", on="lean", swing="z", angle=twist, phase=90)


def limbs(R, arm, leg, armL, armR, legL=None, legR=None, arm_angle=40, leg_angle=32):
    """Arms and legs swinging with the stride, each arm with the other side's leg. Their meshes, by name."""
    out = {}
    for name, pivot, phase in (("armL", armL, 180), ("armR", armR, 0)):
        out[name] = R.group(name, on="body", pivot=pivot, swing="y", angle=arm_angle, phase=phase)
    if legL is not None:
        for name, pivot, phase in (("footL", legL, 0), ("footR", legR, 180)):
            out[name] = R.group(name, on="body", pivot=pivot, swing="y", angle=leg_angle, phase=phase)
    return out


# ----------------------------------------------------------------------------------------- moves the four share
def elevation(pivot, hand):
    """An arm's angle above level, seen from in front: from its pivot to its hand (right side)."""
    return math.degrees(math.atan2(hand[2] - pivot[2], hand[1] - pivot[1]))


def common_moves(R, feet, has_head):
    """Moves every character has: looking around, a hop, a landing squash, a jump with arms up, and flailing in the air
    (feet kicking if it has feet)."""
    look = "head" if has_head else "body"
    mv = R.move("lookaround", "idle", weight=1.2)
    mv.rest(0, look).key(0.6, look, (0, 40, 0)).key(1.3, look, (0, 40, 0)).key(2.0, look, (0, -40, 0))
    mv.key(2.7, look, (0, -40, 0)).rest(3.3, look)
    if has_head:
        mv.rest(0, "body").key(0.6, "body", (0, 10, 0)).key(2.0, "body", (0, -10, 0)).rest(3.3, "body")

    mv = R.move("hop", "idle", weight=0.7)
    mv.rest(0, "lean", "armL", "armR")
    mv.key(0.15, "lean", at=(0, 0, -3), swing=0).key(0.4, "lean", at=(0, 0, 16), swing=0)
    mv.key(0.65, "lean", at=(0, 0, 0), swing=0).key(0.75, "lean", at=(0, 0, -2.5), swing=0).rest(0.95, "lean")
    mv.arm(0.4, "armL", 115).arm(0.4, "armR", 115).rest(0.8, "armL", "armR")

    mv = R.move("squash", "land", blend=0.08)
    mv.rest(0, "lean", "armL", "armR")
    mv.key(0.1, "lean", (-6, 0, 0), (0, 0, -6), swing=0).key(0.35, "lean", at=(0, 0, 2), swing=0).rest(0.55, "lean")
    mv.arm(0.1, "armL", 100).arm(0.1, "armR", 100).rest(0.5, "armL", "armR")

    mv = R.move("cheer", "jump", blend=0.1)
    mv.rest(0, "armL", "armR", "lean")
    mv.arm(0.2, "armL", 165).arm(0.2, "armR", 165)
    mv.arm(0.6, "armL", 155).arm(0.6, "armR", 155).rest(0.9, "armL", "armR")
    mv.key(0.25, "lean", (8, 0, 0)).rest(0.9, "lean")

    mv = R.move("flail", "air", blend=0.2)
    mv.rest(0, "armL", "armR")
    for i, t in enumerate(np.linspace(0.15, 1.35, 7)):
        f = 35 * (1 - 2 * (i % 2))
        mv.arm(t, "armL", 150 - 45 * (i % 2), fwd=f).arm(t, "armR", 105 + 45 * (i % 2), fwd=-f)
    mv.rest(1.5, "armL", "armR")
    if feet:
        mv.rest(0, "footL", "footR").wave("footL", 0.1, 1.4, 9, (35, 0, 0), (-25, 0, 0), swing=0)
        mv.wave("footR", 0.1, 1.4, 9, (-25, 0, 0), (35, 0, 0), swing=0).rest(1.5, "footL", "footR")


def spin(R, name, when, turns=1, length=1.1, hop=10, arms=105, weight=1.0, crouch=0):
    """A spin on the spot (the whole character about the up axis): no easing at the ends, so the full turns don't
    unwind as the move fades; every key starts and ends at rest."""
    mv = R.move(name, when, blend=0, weight=weight)
    mv.rest(0, "lean").key(length * 0.15, "lean", (0, 40 * turns, 0), (0, 0, hop * 0.6 - crouch))
    mv.key(length * 0.5, "lean", (0, 180 * turns, 0), (0, 0, hop - crouch))
    mv.key(length * 0.85, "lean", (0, 320 * turns, 0), (0, 0, hop * 0.5 - crouch)).key(length, "lean", (0, 360 * turns, 0))
    mv.rest(0, "armL", "armR").arm(length * 0.3, "armL", arms)
    mv.arm(length * 0.3, "armR", arms).arm(length * 0.7, "armL", arms)
    mv.arm(length * 0.7, "armR", arms).rest(length, "armL", "armR")
    return mv


def wave_hand(R, name, when, look, weight=1.0, length=1.9):
    """The right arm up and waving, the character looking a little to that side."""
    mv = R.move(name, when, weight=weight)
    mv.rest(0, "armR").arm(0.35, "armR", 125, fwd=10)
    mv.arm_wave("armR", 0.55, length - 0.45, 6, 165, 125, fwd=10)
    mv.rest(length, "armR")
    mv.rest(0, look).key(0.4, look, (6, 18, -6)).key(length - 0.4, look, (6, 18, -6)).rest(length, look)
    return mv


# ------------------------------------------------------------------------------------------------------- Mario
def mario():
    R = Rig("mario", {
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
        "shoe": {"colour": "#6a3416", "rough": 0.45},
    }, TEMPO)
    m = skeleton(R)
    ellipsoid(m, "skin", (0, 0, 0), (BALL, BALL, BALL), nu=16, nv=28)

    # the cap: a puffy dome sitting back on the head, its front higher than its back; it can be tipped (about its back)
    c = R.group("cap", on="body", pivot=(-26, 0, 43))

    def cap_thick(s, phi):
        f, t = lip(s, 0.93, -1.0)
        puff = 4.2 + 3.6 * (1 + np.cos(phi)) / 2 * (1 - s ** 2)
        return puff * f + t
    cap = Patch(d(67.5, 180), 72.5, cap_thick, front=(1, 0, 0))
    cap.build(c, "cap", ns=12, nphi=36)
    # the brim, flat and wide over the eyes, tipped down a little
    pitch = math.radians(18)
    fwd = np.array([math.cos(pitch), 0, -math.sin(pitch)])
    upv = np.array([math.sin(pitch), 0, math.cos(pitch)])
    ellipsoid(c, "cap", (36, 0, 30), (27, 19, 2.6), np.array([[0, 1.0, 0], fwd, upv]), nu=8, nv=28)
    # the emblem: a white circle on the front of the cap with a red M
    E = d(58, 0)
    emb = Patch(E, 17, lambda s, phi: 0.45 * lip(s, 0.85, -1.6)[0] + lip(s, 0.85, -1.6)[1], base=cap.radius_at)
    emb.build(c, "white", ns=8, nphi=36)
    M = np.array([(-1, -1), (-0.6, -1), (-0.6, 0.38), (0, -0.32), (0.6, 0.38), (0.6, -1), (1, -1), (1, 1),
                  (0.62, 1), (0, 0.3), (-0.62, 1), (-1, 1)]) * np.array([8.4, 7.4])
    extrude(c, "cap", M, E * (cap.radius_at(E) + 0.45), frame(E), -1.2, 0.7)

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
            ellipsoid(m, "tache", n * 51.5, (w, h, 4.6), frame(n), nu=6, nv=10)
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

    # white gloves floating at his sides, and brown shoes peeking out in front: they swing as he runs
    L = limbs(R, None, None, (0, -32, -6), (0, 32, -6), (24, -19, 8), (24, 19, 8))
    R.arm_rest, R.arm_float = elevation((0, 32, -6), (6, 59, -14)), (16, 22)
    for s, k in ((-1, "armL"), (1, "armR")):
        glove(L[k], "white", (6, 59 * s, -14), s)
    for s, k in ((-1, "footL"), (1, "footR")):
        ellipsoid(L[k], "shoe", (24, 19 * s, -41), (13.5, 8.5, 6.5), rot((0, 0, 1), 8 * s).T, nu=8, nv=16)

    # ------------------------------------------------------------------ moves
    common_moves(R, feet=True, has_head=False)
    # rolling: a jump with his fist up, a wave, a spin, tipping his cap, a look back over his shoulder
    mv = R.move("fistjump", "run", weight=1.3)
    mv.rest(0, "lean", "armR", "armL", "footL", "footR", "body")
    mv.key(0.15, "lean", at=(0, 0, -4), swing=0).key(0.45, "lean", (6, 0, 0), (0, 0, 26), swing=0)
    mv.key(0.75, "lean", (4, 0, 0), (0, 0, 20), swing=0).key(1.0, "lean", at=(0, 0, -3), swing=0).rest(1.2, "lean")
    mv.arm(0.3, "armR", 170, at=(0, 0, 6)).arm(0.85, "armR", 170, at=(0, 0, 8))
    mv.rest(1.15, "armR").arm(0.45, "armL", 20, fwd=-20).rest(1.0, "armL")
    mv.key(0.45, "footL", (-25, 0, 0), swing=0).key(0.45, "footR", (30, 0, 0), swing=0).rest(1.0, "footL", "footR")
    mv.key(0.45, "body", (0, 12, 0)).rest(1.1, "body")
    wave_hand(R, "wave", "run", "body")
    spin(R, "spin", "run", hop=12, arms=110)
    mv = R.move("captip", "run", weight=0.8)
    mv.rest(0, "armR", "cap").arm(0.4, "armR", 150, fwd=25, at=(6, -14, 14))
    mv.key(0.6, "cap", (18, 0, 0), (0, 0, 3)).key(1.1, "cap", (18, 0, 0), (0, 0, 3))
    mv.arm(1.1, "armR", 150, fwd=25, at=(6, -14, 14)).rest(1.5, "armR", "cap")
    mv = R.move("lookback", "run", weight=0.7)
    mv.rest(0, "body").key(0.45, "body", (0, -100, 4)).key(1.1, "body", (0, -100, 4)).rest(1.6, "body")
    # still: a wave, a tap of his foot, a stretch, a tip of his cap
    wave_hand(R, "hello", "idle", "body")
    mv = R.move("tap", "idle")
    mv.rest(0, "footR").wave("footR", 0.25, 1.75, 8, (10, 0, 0), (0, 0, 0), (0, 0, 4), (0, 0, 0), swing=0)
    mv.rest(2.0, "footR")
    mv.rest(0, "armL", "armR").key(0.4, "armL", (-20, 0, 15), (0, 6, 0), swing=0).key(0.4, "armR", (-20, 0, -15), (0, -6, 0), swing=0)
    mv.key(1.6, "armL", (-20, 0, 15), (0, 6, 0), swing=0).key(1.6, "armR", (-20, 0, -15), (0, -6, 0), swing=0)
    mv.rest(2.0, "armL", "armR")
    mv = R.move("stretch", "idle", weight=0.8)
    mv.rest(0, "armL", "armR", "lean").arm(0.7, "armL", 170)
    mv.arm(0.7, "armR", 170).arm(1.5, "armL", 178, at=(0, 0, 4))
    mv.arm(1.5, "armR", 178, at=(0, 0, 4)).rest(2.3, "armL", "armR")
    mv.key(0.8, "lean", (8, 0, 0), (0, 0, 3)).key(1.5, "lean", (8, 0, 0), (0, 0, 3)).rest(2.3, "lean")
    return R


# ------------------------------------------------------------------------------------------------------- Peach
def peach():
    R = Rig("peach", {
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
    }, TEMPO)
    m = skeleton(R, lean=8, bob=2.5, twist=6)
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

    # the crown, on top and tipped a little forward: a gold band with points, a ruby at the front and sapphires. It
    # wobbles on her head (about its base).
    n = d(70, 0)
    base = n * 55.0
    cr = R.group("crown", on="body", pivot=tuple(base), swing="x", angle=4, phase=45)
    ax = axes_z(n, ref=(1, 0, 0))
    k = 1.3
    lathe(cr, "gold", np.array([(10.5, -2), (12.2, -2), (13.6, 6), (12.4, 6.6), (11.0, 5.8), (9.6, -1)]) * k, base, ax,
          nv=30, closed=True, smooth=False)
    for i in range(5):
        phi = math.radians(i * 72)
        out = math.cos(phi) * ax[0] + math.sin(phi) * ax[1]
        side = np.cross(ax[2], out)
        pc = base + ax[2] * 5.5 * k + out * 12.9 * k
        tri = np.array([(-4.6, 0), (4.6, 0), (0, 8.5)]) * k
        extrude(cr, "gold", tri, pc, np.array([side, ax[2], out]), -0.8, 0.8)
        ellipsoid(cr, "gold", pc + ax[2] * 9.4 * k, (2.0, 2.0, 2.0), nu=5, nv=8)
        gem = base + ax[2] * 2.2 * k + out * 13.6 * k
        if i == 0:
            ellipsoid(cr, "ruby", gem, (3.6, 4.4, 2.0), np.array([side, ax[2], out]), nu=6, nv=12)
        else:
            ellipsoid(cr, "sapphire", gem, (2.5, 2.5, 1.6), np.array([side, ax[2], out]), nu=5, nv=10)
    ellipsoid(cr, "ruby", base + ax[2] * 4.0 * k, (8.5 * k, 8.5 * k, 6.0 * k), ax, nu=8, nv=14)   # the velvet inside

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

    # blue earrings hanging below the hair, swinging from her ears
    for s, name in ((-1, "earL"), (1, "earR")):
        ne = d(-30, 72 * s)
        er = R.group(name, on="body", pivot=tuple(ne * 51.5), swing="y", angle=18, phase=90 + 90 * s)
        ellipsoid(er, "gold", ne * 51.5, (1.3, 1.3, 1.3), nu=4, nv=8)
        ellipsoid(er, "sapphire", d(-36, 73 * s) * 52.5, (3.2, 3.2, 3.2), nu=7, nv=12)

    # her pink dress round the bottom: a darker pink trim and a blue brooch at the front
    dress = Patch((0, 0, -1), 38, lambda s, phi: 2.4 * band(s, 0.42)[0] + band(s, 0.42)[1], s0=0.42)
    dress.build(m, "dress", ns=8, nphi=40)
    trim = Patch((0, 0, -1), 39, lambda s, phi: 3.0 * band(s, 0.86, 0.06)[0] + band(s, 0.86, 0.06)[1], s0=0.86)
    trim.build(m, "trim", ns=4, nphi=40)
    nb = d(-57, 0)
    ellipsoid(m, "gold", nb * 51.8, (6.2, 5.0, 1.8), frame(nb), nu=6, nv=14)
    ellipsoid(m, "sapphire", nb * 53.2, (4.8, 3.8, 1.6), frame(nb), nu=6, nv=14)

    # long white gloves floating at her sides, swinging as she goes
    L = limbs(R, None, None, (0, -32, -8), (0, 32, -8), arm_angle=32)
    R.arm_rest, R.arm_float = elevation((0, 32, -8), (4, 58, -18)), (16, 22)
    for s, k2 in ((-1, "armL"), (1, "armR")):
        glove(L[k2], "white", (4, 58 * s, -18), s, big=0.9)
        ring_y(L[k2], "white", (3, 51.5 * s, -16.5), s, 4.6, 5.6, 3.2)

    # ------------------------------------------------------------------ moves
    common_moves(R, feet=False, has_head=False)
    # rolling: a twirl, a wave, a blown kiss, a hair toss, a happy skip
    spin(R, "twirl", "run", hop=8, arms=120, weight=1.4, length=1.3)
    wave_hand(R, "wave", "run", "body")
    mv = R.move("kiss", "run", weight=0.8)
    mv.rest(0, "armR", "body").arm(0.4, "armR", 50, fwd=75, at=(40, -40, 0))      # a hand to her lips (searched)
    mv.arm(0.7, "armR", 50, fwd=75, at=(40, -40, 0)).arm(1.05, "armR", 70, fwd=75, at=(50, -30, 10))
    mv.rest(1.5, "armR").key(0.6, "body", (-4, 10, 8)).rest(1.4, "body")
    mv = R.move("toss", "run", weight=0.8)
    mv.rest(0, "body", "crown").key(0.3, "body", (4, 0, 14)).key(0.7, "body", (2, 0, -10)).rest(1.2, "body")
    mv.key(0.45, "crown", (0, 0, -10)).key(0.8, "crown", (0, 0, 8)).rest(1.2, "crown")
    mv = R.move("skip", "run", weight=0.8)
    mv.rest(0, "lean", "armL", "armR")
    for i, t in enumerate((0.3, 0.75)):
        mv.key(t, "lean", (0, 0, 8 * (1 - 2 * i)), (0, 0, 12), swing=0).key(t + 0.22, "lean", at=(0, 0, 0), swing=0)
    mv.rest(1.2, "lean").arm(0.3, "armL", 120).arm(0.3, "armR", 120)
    mv.arm(0.9, "armL", 120).arm(0.9, "armR", 120).rest(1.2, "armL", "armR")
    # still: a curtsy, a wave, fixing her crown, a sway
    curtsy = R.move("curtsy", "idle", weight=1.3)
    curtsy.rest(0, "lean", "armL", "armR").key(0.6, "lean", (-12, 0, 0), (0, 0, -3), swing=0)
    curtsy.key(1.3, "lean", (-12, 0, 0), (0, 0, -3), swing=0).rest(2.0, "lean")
    curtsy.key(0.5, "armL", (-25, 0, 28), (0, 4, 0), swing=0).key(0.5, "armR", (-25, 0, -28), (0, -4, 0), swing=0)
    curtsy.key(1.4, "armL", (-25, 0, 28), (0, 4, 0), swing=0).key(1.4, "armR", (-25, 0, -28), (0, -4, 0), swing=0)
    curtsy.rest(2.0, "armL", "armR")
    land = R.move("curtsy2", "land", blend=0.1)
    land.rest(0, "lean").key(0.3, "lean", (-9, 0, 0), (0, 0, -2), swing=0).rest(0.8, "lean")
    land.rest(0, "armL", "armR").key(0.3, "armL", (-20, 0, 25), swing=0).key(0.3, "armR", (-20, 0, -25), swing=0)
    land.rest(0.8, "armL", "armR")
    wave_hand(R, "hello", "idle", "body")
    mv = R.move("crownfix", "idle", weight=0.7)
    mv.rest(0, "armR", "crown").arm(0.5, "armR", 160, fwd=20, at=(6, -14, 16))
    mv.arm(1.2, "armR", 160, fwd=20, at=(6, -14, 16)).rest(1.7, "armR")
    mv.key(0.6, "crown", (0, 0, -9)).key(0.9, "crown", (0, 0, 4)).rest(1.2, "crown")
    mv = R.move("sway", "idle", weight=0.8)
    mv.rest(0, "lean").wave("lean", 0.5, 2.5, 4, (0, 0, 7), (0, 0, -7)).rest(3.0, "lean")
    spin(R, "float", "air", arms=125, hop=0, length=1.6)
    return R


# ------------------------------------------------------------------------------------------------------- Yoshi
def yoshi():
    R = Rig("yoshi", {
        "green": {"colour": "#46b92c", "rough": 0.5},
        "white": {"colour": "#fbfbfb", "rough": 0.45},
        "saddle": {"colour": "#e8261c", "rough": 0.35},
        "spike": {"colour": "#f27a14", "rough": 0.45},
        "shoe": {"colour": "#f2661a", "rough": 0.35},
        "pupil": {"colour": "#0c0c0c", "rough": 0.2},
        "nostril": {"colour": "#1e5a12", "rough": 0.6},
        "tongue": {"colour": "#f0487a", "rough": 0.35},
    }, TEMPO)
    m = skeleton(R, lean=12, bob=3.5, twist=8)
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

    # the head, up on the front of the body: a big round snout, a white chin, eyes on top. It nods about his neck.
    h = R.group("head", on="body", pivot=(12, 0, 36), swing="x", angle=5, phase=90)
    H = np.array([16.0, 0, 54.0])
    ellipsoid(h, "green", H, (25, 24, 25), nu=14, nv=24)
    ellipsoid(h, "green", (44, 0, 60), (22, 21, 18), nu=14, nv=24)               # the snout
    for s in (-1, 1):
        ellipsoid(h, "nostril", (62.5, 6.5 * s, 66), (1.8, 2.4, 3.0), nu=5, nv=8)
    ellipsoid(h, "saddle", (40, 0, 44), (13, 15, 3.2), nu=6, nv=16)              # his mouth
    ellipsoid(h, "white", (35, 0, 47.5), (21, 21.5, 7.5), nu=8, nv=18)          # cheeks and chin
    for s in (-1, 1):
        ec = np.array([24.0, 9.5 * s, 82.0])
        tilt = rot((0, 1, 0), -12).T
        ellipsoid(h, "white", ec, (8.5, 8.0, 12.5), tilt, nu=10, nv=18)
        ellipsoid(h, "pupil", ec + np.array([7.4, 0.8 * s, 0.5]), (2.2, 3.6, 6.6), tilt, nu=6, nv=12)
        ellipsoid(h, "white", ec + np.array([8.6, 0.6 * s, 3.2]), (0.8, 1.2, 1.6), nu=4, nv=8)
    # spikes on the back of his head
    for x, z, up in ((0.0, 74.0, 0.6), (-7.0, 63.0, 0.15)):
        cone(h, "spike", np.array([x, 0, z]), unit(np.array([-1.0, 0, up])), 5.8, 10.0, nv=10)
    # his long tongue, hidden in his snout until he sticks it out (see TONGUE_HIDDEN)
    tg = R.group("tongue", on="head", pivot=(30, 0, 50.5))
    tube(tg, "tongue", np.array([(27.0, 0, 50.5), (40, 0, 50.5), (52, 0, 50.5)]), np.array([3.0, 3.0, 2.8]), nv=10)
    ellipsoid(tg, "tongue", (53.5, 0, 50.5), (4.2, 4.6, 3.0), nu=6, nv=12)

    # little arms, red-orange shoes peeking out in front, and a little tail behind that wags
    L = limbs(R, None, None, (0, -36, 8), (0, 36, 8), (30, -22, 10), (30, 22, 10), arm_angle=45, leg_angle=30)
    R.arm_rest, R.arm_float = elevation((0, 36, 8), (14, 49, -2)), (12, 10)
    for s, k in ((-1, "armL"), (1, "armR")):
        ellipsoid(L[k], "green", (14, 49 * s, -2), (7.0, 6.0, 9.0), rot((1, 0, 0), -20 * s).T, nu=8, nv=14)
        ellipsoid(L[k], "green", (19, 51.5 * s, -8), (4.5, 4.0, 4.0), nu=6, nv=10)
    for s, k in ((-1, "footL"), (1, "footR")):
        ellipsoid(L[k], "shoe", (30, 22 * s, -38), (16, 10.5, 9), rot((0, 0, 1), 12 * s).T, nu=8, nv=16)
    t = R.group("tail", on="body", pivot=(-44, 0, -20), swing="z", angle=22, phase=90)
    tube(t, "green", np.array([(-46.0, 0, -22), (-56, 0, -28), (-64, 0, -30)]), np.array([10, 6, 0.5]), nv=12)
    # his egg (white with green spots), laid from behind him at every landing: it pops out backward, rolls to a stop,
    # lies there 4 seconds and shrinks away; at most 5 at once (host 0.27.0 "drop")
    egg_at = np.array([-58.0, 0, -30])
    e = R.group("egg", on="body", pivot=tuple(egg_at), drop="land", push=(-160, 0, 380), life=4, most=5, carry=0.85)
    ellipsoid(e, "white", egg_at, (11.0, 11.0, 13.5), nu=10, nv=18)
    for el, az, r in ((35, 20, 4.2), (10, 100, 4.6), (-20, 200, 4.4), (15, 250, 3.8), (-35, 320, 3.6), (50, 160, 3.4),
                      (-5, 30, 3.2), (-50, 120, 3.4)):
        n = d(el, az)
        c = egg_at + n * np.array([11.0, 11.0, 13.5])
        ellipsoid(e, "green", c, (r, r, 1.0), frame(unit(n / np.array([11.0, 11.0, 13.5]) ** 2)), nu=5, nv=10)

    # ------------------------------------------------------------------ moves
    common_moves(R, feet=True, has_head=True)
    # flutter kick: legs paddling fast, a float up, arms out (rolling, jumping and in the air)
    for name, when, length, w in (("flutter", "run", 1.6, 1.3), ("flutterjump", "jump", 1.1, 1.0),
                                  ("flutterair", "air", 1.6, 2.0)):
        mv = R.move(name, when, weight=w, blend=0.15)
        mv.rest(0, "lean", "footL", "footR", "armL", "armR", "head")
        mv.key(0.3, "lean", (6, 0, 0), (0, 0, 14 if when == "run" else 4), swing=0)
        mv.key(length - 0.35, "lean", (6, 0, 0), (0, 0, 16 if when == "run" else 4), swing=0).rest(length, "lean")
        n = int(length * 9) | 1
        mv.wave("footL", 0.12, length - 0.15, n, (45, 0, 0), (-20, 0, 0), swing=0)
        mv.wave("footR", 0.12, length - 0.15, n, (-20, 0, 0), (45, 0, 0), swing=0).rest(length, "footL", "footR")
        mv.arm(0.3, "armL", 110).arm(0.3, "armR", 110)
        mv.arm(length - 0.3, "armL", 110).arm(length - 0.3, "armR", 110)
        mv.rest(length, "armL", "armR").key(0.4, "head", (14, 0, 0)).key(length - 0.4, "head", (14, 0, 0)).rest(length, "head")
    # his tongue: shot out and pulled back in
    for name, when, w in (("tongue", "run", 1.1), ("lick", "idle", 1.0)):
        mv = R.move(name, when, weight=w, blend=0.15)
        mv.rest(0, "head", "tongue").key(0.25, "head", (-6, 0, 0)).key(0.55, "head", (-10, 0, 0)).rest(1.2, "head")
        mv.key(0.3, "tongue", at=(2, 0, 0)).key(0.5, "tongue", (-8, 0, 0), (40, 0, -2)).key(0.65, "tongue", (-8, 0, 0), (40, 0, -2))
        mv.key(0.85, "tongue", at=(2, 0, 0)).rest(0.95, "tongue")
    # a big tail wag, with his body
    for name, when in (("wag", "run"), ("wagidle", "idle")):
        mv = R.move(name, when)
        mv.rest(0, "tail", "body").wave("tail", 0.2, 1.5, 8, (0, 40, 0), (0, -40, 0), swing=0).rest(1.7, "tail")
        mv.wave("body", 0.2, 1.5, 8, (0, -6, 0), (0, 6, 0)).rest(1.7, "body")
    # "Yoshi!": head thrown back, arms up
    mv = R.move("yoshi", "run", weight=0.9)
    mv.rest(0, "head", "armL", "armR").key(0.3, "head", (22, 0, 0)).key(0.75, "head", (22, 0, 0)).rest(1.1, "head")
    mv.arm(0.3, "armL", 155).arm(0.3, "armR", 155)
    mv.arm(0.75, "armL", 145).arm(0.75, "armR", 145).rest(1.1, "armL", "armR")
    spin(R, "spin", "run", hop=10, arms=110, weight=0.6)
    # still: a stretch, a tap of his shoe
    mv = R.move("stretch", "idle", weight=0.8)
    mv.rest(0, "armL", "armR", "head").arm(0.7, "armL", 170).arm(0.7, "armR", 170)
    mv.arm(1.5, "armL", 178).arm(1.5, "armR", 178).rest(2.2, "armL", "armR")
    mv.key(0.8, "head", (18, 0, 0)).key(1.5, "head", (18, 0, 0)).rest(2.2, "head")
    mv = R.move("tap", "idle", weight=0.8)
    mv.rest(0, "footR").wave("footR", 0.25, 1.75, 8, (-10, 0, 0), (0, 0, 0), (0, 0, 5), (0, 0, 0), swing=0)
    mv.rest(2.0, "footR")
    return R


# ------------------------------------------------------------------------------------------------------ Bowser
def bowser():
    R = Rig("bowser", {
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
    }, TEMPO)
    m = skeleton(R, lean=9, bob=3.5, twist=9)
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

    # the head, up on the front: a wide snout, fangs, red eyes, angry brows, horns and a red mane. It nods about his
    # neck; his cream jaw (with the lower fangs) opens about its hinge.
    h = R.group("head", on="body", pivot=(8, 0, 40), swing="y", angle=4, phase=90)
    ellipsoid(h, "skin", (18.0, 0, 56.0), (24, 25, 23), nu=12, nv=20)
    ellipsoid(h, "skin", (38, 0, 57), (18, 20, 12.5), nu=10, nv=18)                # snout
    ellipsoid(h, "mouth", (42, 0, 50.5), (10, 16, 3.0), nu=6, nv=16)
    jw = R.group("jaw", on="head", pivot=(22, 0, 50))
    ellipsoid(jw, "belly", (36, 0, 46), (16, 18, 7.5), nu=8, nv=20)                # jaw
    for s in (-1, 1):
        cone(h, "white", np.array([49.0, 8.5 * s, 52.0]), (0.15, 0, -1), 2.2, 6.0, nv=8)    # fangs
        cone(jw, "white", np.array([47.0, 13.0 * s, 47.0]), (0.15, 0, 1), 1.8, 4.5, nv=8)
        ellipsoid(h, "nostril", (54.5, 6.0 * s, 62), (1.4, 2.4, 1.6), nu=4, nv=8)
        # eyes
        ec = np.array([33.5, 10.5 * s, 67.0])
        n = unit(np.array([0.85, 0.25 * s, 0.35]))
        ax = frame(n)
        ellipsoid(h, "white", ec, (4.6, 6.2, 3.0), ax, nu=7, nv=14)
        ic = ec + n * 2.0 + ax[1] * -0.6
        ellipsoid(h, "iris", ic, (3.2, 4.4, 1.4), ax, nu=6, nv=12)
        ellipsoid(h, "pupil", ic + n * 0.7, (1.4, 2.6, 0.9), ax, nu=5, nv=10)
        # brows: thick red, slanting down to the middle
        brow = np.array([(37.5, 4.0 * s, 71.0), (36.0, 10.0 * s, 74.5), (32.5, 16.0 * s, 77.5)])
        tube(h, "hair", brow, np.array([2.2, 2.8, 1.6]), nv=8)
        # horns: cream, out to the sides and curving up
        horn = np.array([(16.0, 18.0 * s, 70.0), (14.0, 27.0 * s, 77.0), (11.0, 32.0 * s, 87.0), (6.0, 32.0 * s, 96.0)])
        t = np.linspace(0, 1, 9)
        path = np.array([np.interp(t, np.linspace(0, 1, 4), horn[:, c]) for c in range(3)]).T
        tube(h, "horn", path, np.linspace(6.5, 0.2, len(path)), nv=10)
    # the mane: red spikes standing up and back over the head
    for p, dirn, r, hh in (((10, 0, 76), (0.25, 0, 1), 6.0, 15), ((4, -9, 72), (-0.4, -0.6, 1), 5.0, 12),
                           ((4, 9, 72), (-0.4, 0.6, 1), 5.0, 12), ((0, 0, 70), (-0.8, 0, 0.8), 6.0, 15),
                           ((-4, -8, 62), (-1, -0.5, 0.4), 5.0, 12), ((-4, 8, 62), (-1, 0.5, 0.4), 5.0, 12),
                           ((-6, 0, 56), (-1, 0, 0.1), 5.0, 12)):
        cone(h, "hair", np.array(p, float), unit(dirn), r, hh, nv=10)

    # arms with black spiked cuffs and claws (swinging from the shoulder), and feet with claws
    L = limbs(R, None, None, (4, -40, 16), (4, 40, 16), (30, -24, 5), (30, 24, 5), arm_angle=35, leg_angle=28)
    R.arm_rest, R.arm_float = elevation((4, 40, 16), (22, 50, -20)), (0, 0)
    for s, k in ((-1, "armL"), (1, "armR")):
        a = L[k]
        arm = np.array([(4.0, 40.0 * s, 16.0), (12.0, 50.0 * s, -2.0), (20.0, 50.0 * s, -16.0)])
        tube(a, "skin", arm, np.array([10.0, 9.5, 9.0]), nv=12)
        ellipsoid(a, "skin", (22, 50 * s, -20), (9.5, 9, 8.5), nu=8, nv=14)
        wrist = np.array([16.0, 50.5 * s, -9.0])
        wn = unit(np.array([0.5, 0.0, -1.0]))
        wa = axes_z(wn)
        lathe(a, "cuff", [(10.2, -2.6), (11.4, -1.6), (11.4, 1.6), (10.2, 2.6), (9.0, 0)], wrist, wa, nv=20,
              closed=True)
        for kk in range(5):
            ang = math.radians(kk * 72 + 18)
            out = math.cos(ang) * wa[0] + math.sin(ang) * wa[1]
            cone(a, "spike", wrist + out * 10.8, out, 2.6, 6.5, nv=8)
        for s2 in (-1, 0, 1):
            cone(a, "spike", np.array([29.0, 50 * s + 4.5 * s2, -23.0]), (1, 0.1 * s2, -0.3), 1.6, 4.5, nv=6)
    for s, k in ((-1, "footL"), (1, "footR")):
        f = L[k]
        ellipsoid(f, "skin", (30, 24 * s, -40), (15, 11, 8), nu=8, nv=16)
        for s2 in (-1, 0, 1):
            cone(f, "spike", np.array([43.0, 24 * s + 5 * s2, -42.0]), (1, 0.15 * s2, -0.15), 1.8, 5.0, nv=6)

    # ------------------------------------------------------------------ moves
    common_moves(R, feet=True, has_head=True)
    # a roar: head thrown back, jaw wide, arms up and out (rolling and still)
    for name, when, w in (("roar", "run", 1.3), ("roaridle", "idle", 1.3)):
        mv = R.move(name, when, weight=w)
        mv.rest(0, "head", "jaw", "armL", "armR", "lean")
        mv.key(0.35, "head", (24, 0, 0), swing=0).key(1.3, "head", (20, 0, 0), swing=0).rest(1.8, "head")
        mv.key(0.4, "jaw", (-30, 0, 0)).wave("jaw", 0.6, 1.2, 4, (-34, 0, 0), (-26, 0, 0)).rest(1.6, "jaw")
        mv.key(0.4, "armL", (25, 0, 100), swing=0).key(0.4, "armR", (25, 0, -100), swing=0)
        mv.key(1.3, "armL", (25, 0, 95), swing=0).key(1.3, "armR", (25, 0, -95), swing=0).rest(1.8, "armL", "armR")
        mv.key(0.4, "lean", (7, 0, 0), swing=0).key(1.3, "lean", (6, 0, 0), swing=0).rest(1.8, "lean")
    # a stomp: one foot up and slammed down, then the other, the whole of him thumping
    for name, when in (("stomp", "run"), ("stompidle", "idle")):
        mv = R.move(name, when, weight=1.2)
        mv.rest(0, "lean", "armL", "armR")
        for i, (t, foot) in enumerate(((0.3, "footR"), (0.85, "footL"))):
            mv.rest(t - 0.3, foot, swing=0).key(t, foot, (25, 0, 0), (0, 0, 16), swing=0).key(t + 0.15, foot, swing=0)
            mv.key(t, "lean", (0, 0, 6 * (1 - 2 * i)), (0, 0, 7), swing=0).key(t + 0.15, "lean", at=(0, 0, -4), swing=0)
            mv.key(t + 0.3, "lean", at=(0, 0, 0), swing=0)
        mv.rest(1.4, "footL", "footR", "lean")
        mv.key(0.3, "armL", (0, 0, 40), swing=0).key(0.3, "armR", (0, 0, -40), swing=0)
        mv.key(1.1, "armL", (0, 0, 40), swing=0).key(1.1, "armR", (0, 0, -40), swing=0).rest(1.4, "armL", "armR")
    # a flex: arms up, pumped twice
    for name, when in (("flex", "run"), ("flexidle", "idle")):
        mv = R.move(name, when)
        mv.rest(0, "armL", "armR", "head")
        mv.wave("armL", 0.35, 1.5, 4, (60, 0, 85), (50, 0, 70), swing=0).rest(1.9, "armL")
        mv.wave("armR", 0.35, 1.5, 4, (60, 0, -85), (50, 0, -70), swing=0).rest(1.9, "armR")
        mv.key(0.4, "head", (-8, 0, 0)).key(1.5, "head", (-8, 0, 0)).rest(1.9, "head")
    # his shell spin: tucked into his shell (head and arms in), spinning twice
    mv = spin(R, "shellspin", "run", turns=2, length=1.3, hop=4, arms=20, weight=1.0, crouch=6)
    mv.rest(0, "head").key(0.25, "head", (-14, 0, 0), (-6, 0, -8)).key(1.05, "head", (-14, 0, 0), (-6, 0, -8)).rest(1.3, "head")
    # a claw swipe with his right arm
    mv = R.move("swipe", "run", weight=0.8)
    mv.rest(0, "armR", "body").key(0.35, "armR", (110, 0, -40), swing=0).key(0.6, "armR", (-30, 0, -10), swing=0)
    mv.rest(1.0, "armR").key(0.35, "body", (0, 18, 0)).key(0.6, "body", (0, -15, 0)).rest(1.0, "body")
    return R


CHARACTERS = {
    "mario": ("Mario", mario, "Mario: red cap with his M, moustache, big nose, blue eyes, brown hair, red shirt and blue overalls, white gloves and brown shoes."),
    "yoshi": ("Yoshi", yoshi, "Yoshi: green with a white belly and chin, big snout, eyes on top, red saddle, orange spikes and shoes."),
    "peach": ("Peach", peach, "Peach: gold crown with a ruby and sapphires, blonde hair, blue eyes, sapphire earrings, pink dress and brooch, white gloves."),
    "bowser": ("Bowser", bowser, "Bowser: spiked green shell with a cream rim, horns, red mane and brows, red eyes, fangs, spiked cuffs."),
}


def check(R):
    """Measured, not assumed: Yoshi's tongue is inside his head at rest (every point of it inside the snout, the head or
    the chin), and nothing pokes far below the ground at rest."""
    if R.key == "yoshi":
        V = np.concatenate([V for items in R.groups["tongue"]["mesh"].parts.values() for V, _, _ in items])
        def inside(c, r):
            return (((V - np.array(c)) / np.array(r)) ** 2).sum(1) < 1
        hidden = inside((44, 0, 60), (22, 21, 18)) | inside((16, 0, 54), (25, 24, 25)) | inside((35, 0, 47.5), (21, 21.5, 7.5))
        assert hidden.all(), f"tongue shows at rest: {np.round(V[~hidden][:5], 1)}"
    lo = min(m.bounds()[0][2] for n, m in R.parts() if not R.groups[n]["drop"])
    assert lo > -BALL - 1.5, f"{R.key} reaches {lo:.1f} below the ball's bottom"


def write(key, sheets=False):
    name, build, desc = CHARACTERS[key]
    R = build()
    check(R)
    for old in MODELS.glob(f"{key}*.glb"):
        old.unlink()
    header = [f"# {desc}",
              f"# Made from scratch by tools/make_models.py ({R.triangles()} triangles in {len(R.parts())} parts). The game's",
              "# clear ball is hidden: the character is the whole ball, its head or body a sphere of the ball's size, kept",
              "# upright and facing where the ball goes. It leans into the run and bounces each step, twists, swings its",
              "# arms and legs, and now and then does a move: picked at random for what the ball is doing (moves, below).",
              "# Needs host 0.27.0 (move, key, lean)."]
    text, parts = R.text(header)
    write_glb_nodes(list(parts.items()), MODELS / f"{key}.glb")
    (MODELS / f"{key}.txt").write_text(text, newline="\n")
    rest = R.posed()
    lo, hi = rest.bounds()
    pic = render(rest, size=256, azimuth=-28, elevation=10, extent=max(66, np.max(np.abs(np.r_[lo, hi])) * 1.02),
                 centre=(0, 0, (lo[2] + hi[2]) / 2))
    pic.save(PREVIEWS / f"{key}.png")
    print(f"{key}: {R.triangles()} triangles in {len(R.parts())} parts, {len(R.moves)} moves, bounds "
          f"{np.round(lo, 1)} .. {np.round(hi, 1)}")
    if sheets:
        sheet_of(R)
    return pic


def sheet_of(R):
    """A picture of every move at its busiest moment, and the stride at both ends, for checking the directions."""
    out = PREVIEWS / "moves"
    out.mkdir(exist_ok=True)
    tiles = [("rest", R.posed()),
             ("stride a", R.posed(turn=math.pi / 2, reach=1, pace_share=1)),
             ("stride b", R.posed(turn=-math.pi / 2, reach=1, pace_share=1))]
    for mv in R.moves:
        # the busiest key time: the one furthest from rest, summed over the groups
        times = sorted({k[0] for k in mv.keys})
        def busy(t):
            return sum(np.abs(r).sum() + 3 * np.abs(a).sum() for r, a, _ in R.at_time(mv, t).values())
        t = max(times, key=busy)
        tiles.append((f"{mv.name} ({mv.when}) {t:.2f}s", R.posed(keyed=R.at_time(mv, t))))
    S = 200
    cols = 5
    img = Image.new("RGBA", (S * cols, (S + 16) * ((len(tiles) + cols - 1) // cols)), (40, 44, 52, 255))
    dr = ImageDraw.Draw(img)
    for i, (label, m) in enumerate(tiles):
        pic = render(m, size=S, ss=2, azimuth=-35, elevation=12, extent=95, centre=(0, 0, 5))
        x, y = (i % cols) * S, (i // cols) * (S + 16)
        img.alpha_composite(pic, (x, y + 16))
        dr.text((x + 4, y + 2), label, fill=(255, 255, 255, 255))
    img.save(out / f"{R.key}.png")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sheets = "--sheets" in sys.argv
    keys = args or list(CHARACTERS)
    MODELS.mkdir(exist_ok=True)
    PREVIEWS.mkdir(exist_ok=True)
    pics = {k: write(k, sheets) for k in keys}
    if len(pics) == len(CHARACTERS):
        icon = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        for i, k in enumerate(CHARACTERS):
            icon.alpha_composite(pics[k], ((i % 2) * 256, (i // 2) * 256))
        icon.save(ROOT / "icon.png")


if __name__ == "__main__":
    main()
