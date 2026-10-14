"""A character as moving parts: groups (each part a .glb of its own, turning about a pivot, built on another group), the
tempo of its run, and its moves (keyframed animations the host plays one after another, picked at random for what the
ball is doing). Written as the model text the plugin manager reads (docs/guides/cosmetics.md, "Models" and "Moves").

Poses are worked out here exactly as the host does (models.cpp Animate), so the tile pictures and the move sheets show
what the game will show: a group's rotation is an Unreal rotator (pitch about y, nose up; yaw about z, +x toward +y;
roll about x, +y down), turned about its pivot, carried by the group it is built on.
"""
import math

import numpy as np

from meshkit import Mesh


def rotator(pitch, yaw, roll):
    """FRotationMatrix as a 3x3 whose columns are where x, y and z go (models.cpp Turn)."""
    k = math.pi / 180
    sp, cp, sy, cy, sr, cr = (math.sin(pitch * k), math.cos(pitch * k), math.sin(yaw * k), math.cos(yaw * k),
                              math.sin(roll * k), math.cos(roll * k))
    x = [cp * cy, cp * sy, sp]
    y = [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp]
    z = [-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp]
    return np.array([x, y, z]).T


def fmt(v):
    return ",".join(f"{x:.4g}" if abs(x) >= 1e-4 else "0" for x in v)


class Move:
    def __init__(self, name, when, length=None, weight=1.0, blend=0.25, rate=1.0, run=0.0, lo=None, hi=None):
        self.name, self.when, self.length, self.weight, self.blend = name, when, length, weight, blend
        self.rate, self.run, self.lo, self.hi = rate, run, lo, hi
        self.keys = []                                  # (t, group, rot, at, swing)

    def key(self, t, group, rot=(0, 0, 0), at=(0, 0, 0), swing=1.0):
        self.keys.append((float(t), group, tuple(map(float, rot)), tuple(map(float, at)), float(swing)))
        return self

    def arm(self, t, group, lift, fwd=0.0, at=(0, 0, 0), swing=0.0):
        """An arm (armL or armR) raised out to the side: `lift` degrees from hanging straight down (90 out level, 180
        straight up), `fwd` tipping it forward. The rig's arm_rest is the arm's own angle at rest (above level, from
        its pivot to its hand); arms that float beside the body (arm_float: cm out and up when straight up) are also
        carried out and up as they rise, so a raised hand clears the head."""
        s = -1 if group == "armL" else 1
        rig = self.rig
        roll = -s * ((lift - 90) - rig.arm_rest)
        f = min(1.0, max(0.0, (lift - 60) / 120))
        out, up = rig.arm_float
        a = (at[0], at[1] + s * out * f, at[2] + up * f)
        return self.key(t, group, (fwd, 0, roll), a, swing)

    def arm_wave(self, group, t0, t1, n, lift_a, lift_b, fwd=0.0, at=(0, 0, 0)):
        for i in range(n):
            self.arm(t0 + (t1 - t0) * i / max(n - 1, 1), group, lift_a if i % 2 == 0 else lift_b, fwd, at)
        return self

    def rest(self, t, *groups, swing=1.0):
        for g in groups:
            self.key(t, g, swing=swing)
        return self

    def wave(self, group, t0, t1, n, a, b, at_a=(0, 0, 0), at_b=None, swing=1.0):
        """n keys from t0 to t1 alternating between poses a and b (rotations), starting on a."""
        at_b = at_a if at_b is None else at_b
        for i in range(n):
            t = t0 + (t1 - t0) * i / max(n - 1, 1)
            self.key(t, group, a if i % 2 == 0 else b, at_a if i % 2 == 0 else at_b, swing)
        return self

    def end(self):
        return self.length if self.length else max(k[0] for k in self.keys)

    def sample(self, group, t):
        """The group's keyed pose at t (linear between keys: enough for the pictures), or None."""
        ks = sorted((k for k in self.keys if k[1] == group), key=lambda k: k[0])
        if not ks:
            return None
        if t <= ks[0][0]:
            return ks[0][2], ks[0][3], ks[0][4]
        for a, b in zip(ks, ks[1:]):
            if a[0] <= t <= b[0]:
                u = (t - a[0]) / max(b[0] - a[0], 1e-9)
                lerp = lambda p, q: tuple(x + (y - x) * u for x, y in zip(p, q))
                return lerp(a[2], b[2]), lerp(a[3], b[3]), a[4] + (b[4] - a[4]) * u
        return ks[-1][2], ks[-1][3], ks[-1][4]

    def text(self):
        opts = [f"when={self.when}"]
        if self.length:
            opts.append(f"length={self.length:g}")
        if self.weight != 1:
            opts.append(f"weight={self.weight:g}")
        if self.blend != 0.25:
            opts.append(f"blend={self.blend:g}")
        if self.rate != 1:
            opts.append(f"rate={self.rate:g}")
        if self.run:
            opts.append(f"run={self.run:g}")
        if self.lo is not None:
            opts.append(f"min={self.lo:g}")
        if self.hi is not None:
            opts.append(f"max={self.hi:g}")
        lines = [f"move {self.name} " + " ".join(opts)]
        for t, g, r, a, s in sorted(self.keys, key=lambda k: (k[1], k[0])):
            w = [f"key {t:g} {g}"]
            if any(abs(x) > 1e-6 for x in r):
                w.append(f"rot={fmt(r)}")
            if any(abs(x) > 1e-6 for x in a):
                w.append(f"at={fmt(a)}")
            if s != 1:
                w.append(f"swing={s:g}")
            lines.append(" ".join(w))
        return lines


class Rig:
    def __init__(self, key, materials, tempo):
        self.key, self.materials, self.tempo = key, materials, tempo
        self.groups = {}                                # name -> dict, in order
        self.moves = []
        self.gap = (0.4, 1.6)
        self.arm_rest = 0.0                             # the arms' angle at rest above level (Move.arm)
        self.arm_float = (0.0, 0.0)

    def group(self, name, on=None, pivot=(0, 0, 0), travel=False, mesh=True, swing=None, angle=0, phase=0, bob=0,
              lean=0, drop=None, push=None, life=None, most=None, carry=None):
        m = Mesh(self.materials) if mesh else None
        self.groups[name] = dict(on=on, pivot=np.array(pivot, float), travel=travel, mesh=m, swing=swing,
                                 angle=angle, phase=phase, bob=bob, lean=lean, drop=drop, push=push, life=life,
                                 most=most, carry=carry)
        return m

    def move(self, *a, **k):
        mv = Move(*a, **k)
        mv.rig = self
        self.moves.append(mv)
        return mv

    def parts(self):
        return [(n, g["mesh"]) for n, g in self.groups.items() if g["mesh"] is not None and g["mesh"].parts]

    def triangles(self):
        return sum(m.triangles() for _, m in self.parts())

    # ----------------------------------------------------------------------------------------------- the pose
    def transforms(self, keyed=None, turn=0.0, reach=0.0, pace_share=0.0):
        """Each group's world transform (R, t: model point p -> R @ (p - pivot) + t), as the host's Animate places
        them: swings at `turn` (radians of the tempo) and `reach`, keyed poses from `keyed` {group: (rot, at, swing)}."""
        out = {}
        for name, g in self.groups.items():
            rot, at, keep = (0, 0, 0), (0, 0, 0), 1.0
            if keyed and name in keyed:
                rot, at, keep = keyed[name]
            angles = [0.0, 0.0, 0.0]                    # x (roll), y (pitch), z (yaw)
            if g["swing"]:
                ax = "xyz".index(g["swing"])
                angles[ax] += g["angle"] * reach * keep * math.sin(turn + math.radians(g["phase"]))
            pitch = angles[1] + rot[0] - g["lean"] * pace_share
            R = rotator(pitch, angles[2] + rot[1], angles[0] + rot[2])
            base = self.groups[g["on"]]["pivot"] if g["on"] else np.zeros(3)
            loc = g["pivot"] - base + np.array(at, float)
            loc[2] += g["bob"] * reach * keep * (0.5 - 0.5 * math.cos(2 * turn + math.radians(g["phase"])))
            if g["on"]:
                PR, Pt = out[g["on"]]
                out[name] = (PR @ R, PR @ loc + Pt)
            else:
                out[name] = (R, loc)
        return out

    def posed(self, keyed=None, turn=0.0, reach=0.0, pace_share=0.0):
        """One mesh of every part in that pose (for pictures)."""
        tf = self.transforms(keyed, turn, reach, pace_share)
        out = Mesh(self.materials)
        for name, m in self.parts():
            if self.groups[name]["drop"]:
                continue                                # not shown on the ball
            R, t = tf[name]
            piv = self.groups[name]["pivot"]
            for mat, items in m.parts.items():
                for V, F, smooth in items:
                    out.parts.setdefault(mat, []).append(((V - piv) @ R.T + t, F, smooth))
        return out

    def at_time(self, move, t):
        return {n: p for n in self.groups if (p := move.sample(n, t)) is not None}

    # ----------------------------------------------------------------------------------------------- the text
    def text(self, header):
        L = list(header)
        tp = self.tempo
        L.append("tempo " + " ".join(f"{k}={v:g}" for k, v in tp.items()))
        L.append("ball hidden")
        files = {}
        for name, g in self.groups.items():
            w = [f"group {name}"]
            if g["travel"]:
                w.append("travel")
            if g["on"]:
                w.append(f"on={g['on']}")
            if np.any(g["pivot"] != 0):
                w.append(f"pivot={fmt(g['pivot'])}")
            if g["swing"]:
                w.append(f"swing={g['swing']} angle={g['angle']:g}")
            if g["bob"]:
                w.append(f"bob={g['bob']:g}")
            if g["phase"] and (g["swing"] or g["bob"]):
                w.append(f"phase={g['phase']:g}")
            if g["lean"]:
                w.append(f"lean={g['lean']:g}")
            if g["drop"]:
                w.append(f"drop={g['drop']}")
                if g["push"] is not None:
                    w.append(f"push={fmt(g['push'])}")
                if g["life"] is not None:
                    w.append(f"life={g['life']:g}")
                if g["most"] is not None:
                    w.append(f"max={g['most']}")
                if g["carry"] is not None:
                    w.append(f"carry={g['carry']:g}")
            L.append(" ".join(w))
            m = g["mesh"]
            if m is not None and m.parts:
                lo, hi = m.bounds()
                c = (lo + hi) / 2
                files[name] = m
                L.append(f"mesh {self.key}.glb node={name} at={c[0]:.3f},{c[1]:.3f},{lo[2]:.3f}")
        L.append("")
        L.append(f"moves gap={self.gap[0]:g},{self.gap[1]:g}")
        for mv in self.moves:
            L.extend(mv.text())
        return "\n".join(L) + "\n", files
