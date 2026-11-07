"""Live car preview for VPK tool: draws a Re3DX+ car pack (.wpk) with the sticker you are editing blended on.

The geometry of a view is rasterised once into a small "g-buffer" (which texture, which texel, how lit, per pixel).
After that, changing the sticker only re-blends the body texture and re-reads the texels, which takes a blink,
so the picture follows every Replace / Colour / Undo straight away.

Software render (opaque first, then see-through parts back to front, like the game); no game lighting or reflections.
The .wpk reader and the drawing are those of idas3_to_wpk.py.
"""
import json
import os
import queue
import re
import struct
import threading
import time

import numpy as np
from PIL import Image

WPK_SHADOW = 0xFFFF
WPK_LIT_CAR = 1        # batch program of the car shader (alpha 1: no cut-out); from the packs, render.c has no value
VERTEX_LIT = np.dtype([("pos", "<f4", 3), ("uv", "<f4", 2), ("rgba", "u1", 4),
                       ("normal", "<i2", 3), ("pad", "<i2")])
VIEWS = [("Front 3/4", -35, 14), ("Side", 90, 0), ("Rear 3/4", 145, 16), ("Rear", 180, 0), ("Front", 0, 0),
         ("Top", 0, 89)]
BG = np.array([0.55, 0.62, 0.70], np.float32)
FREE = len(VIEWS)                     # index of the free-turn view (after the six fixed ones)

# The car pack holds every optional part at once, each variant as its own chunk tagged with a part number.
# Drawn together they overlap (four wings, four bonnets ...), so the preview shows one variant (or none) per group.
# Variant letters follow the order below: Wing A = tag 290, B = 291 ...; Wheels A = 256-259, B = 298-301.
PART_GROUPS = [("Wing", [[290], [291], [292], [293]]),
               ("Bonnet", [[294], [295], [296], [297]]),
               ("Mirrors", [[288], [289]]),
               ("Wheels", [[256, 257, 258, 259], [298, 299, 300, 301]])]
KIT_RE = re.compile(r"^(.+?)(?:_aero([ABC]))?\.wpk$", re.I)     # maz_FC3S_aeroA.wpk -> (maz_FC3S, A)


class WpkError(Exception):
    pass


# ---- settings that are remembered between sessions (window sizes, lighting, reflection, night ...)
PREFS_PATH = os.path.join(os.path.expanduser("~"), ".vpk_tool_settings.json")


def load_prefs():
    try:
        with open(PREFS_PATH, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_prefs(**kv):
    """merge kv into the settings file (never raises: a read-only home folder just means nothing is remembered)"""
    d = load_prefs()
    d.update(kv)
    try:
        with open(PREFS_PATH, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1)
    except OSError:
        pass


# ------------------------------------------------------------------------------------------------
# the file
# ------------------------------------------------------------------------------------------------
def stock_of(path):
    """the stock pack an aero pack borrows its textures from (maz_FC3S_aeroA.wpk -> maz_FC3S.wpk), else None"""
    m = re.match(r"(.+)_aero[ABC]\.wpk$", os.path.basename(path), re.I)
    p = os.path.join(os.path.dirname(os.path.abspath(path)), m.group(1) + ".wpk") if m else None
    return p if p and os.path.isfile(p) else None


def read_wpk3(path, stock=None):
    """-> (textures [(name, RGBA uint8 h x w x 4)], chunks, batches)
    An aero pack leaves the textures it shares with the stock pack empty (size 0, found there by name): pass the
    stock pack's textures as `stock` to fill them in."""

    try:
        with open(path, "rb") as fh:
            b = fh.read()
        magic, nt, nc, nb, _cs, to, co, bo = struct.unpack_from("<4sIIIfIII", b, 0)
    except (OSError, struct.error) as ex:
        raise WpkError("Cannot read %s: %s" % (os.path.basename(path), ex))
    if magic != b"WPK3":
        raise WpkError("%s is not a WPK3 car pack." % os.path.basename(path))
    tex = []
    for i in range(nt):
        nm, w, h, fmt, off, sz = struct.unpack_from("<48sHHIII", b, to + 64 * i)
        name = nm.split(b"\0")[0].decode("latin1")
        if sz == 0:                                     # borrowed from the stock pack, by name
            hit = next((im for n, im in stock or [] if n == name), None)
            if hit is None:
                hit = next((im for n, im in stock or [] if _stem(n) == _stem(name)), None)
            tex.append((name, hit if hit is not None else np.full((max(h, 1), max(w, 1), 4), 128, np.uint8)))
            continue
        px = np.frombuffer(b, "<u2", count=sz // 2, offset=off)
        if px.size != w * h:
            raise WpkError("Texture %d of the pack has an unexpected size." % i)
        px = px.reshape(h, w)
        if fmt == 0:
            r, g, bl, a = (px >> 11) << 3, ((px >> 5) & 63) << 2, (px & 31) << 3, np.full(px.shape, 255)
        elif fmt == 2:
            r, g, bl, a = (px >> 11) << 3, ((px >> 6) & 31) << 3, ((px >> 1) & 31) << 3, (px & 1) * 255
        else:
            r, g, bl, a = (px >> 12) << 4, ((px >> 8) & 15) << 4, ((px >> 4) & 15) << 4, (px & 15) << 4
        tex.append((name, np.stack([r, g, bl, a], -1).astype(np.uint8)))
    chunks = [struct.unpack_from("<6fIII", b, co + 36 * i) for i in range(nc)]
    batches = []
    for i in range(nb):
        t, fl, vo, vn, io, inn = struct.unpack_from("<HHIIII", b, bo + 28 * i)
        V = np.frombuffer(b, VERTEX_LIT, count=vn, offset=vo)
        I = np.frombuffer(b, "<u2", count=inn, offset=io).reshape(-1, 3)
        prog, _pad, env, spec, pw = struct.unpack_from("<BBHHH", b, bo + 28 * i + 20)
        batches.append(dict(tex=t, flags=fl, V=V, I=I, prog=prog,
                            env=env if env < nt else 0xFFFF, spec=spec if spec < nt else 0xFFFF, power=pw / 10000.0))
    return tex, chunks, batches


def _stem(n):
    return os.path.splitext(n)[0].lower()


PAINT = re.compile(r"^p(\d+):(.*)$", re.I)


def find_texture(names, want, paint=0):
    """index of the pack texture a .vpk names (it writes "fc3s_body01_white", the pack has "fc3s_body01_white.dds"):
    the pack's texture `want`, or for paint > 0 its "p<paint>:" copy; None if there is none"""
    if not want:
        return None
    w = _stem(want)
    for i, n in enumerate(names):
        m = PAINT.match(n)
        if paint > 0:
            if m and int(m.group(1)) == paint and _stem(m.group(2)) == w:
                return i
        elif not m and (n == want or _stem(n) == w):
            return i
    if paint > 0:
        return find_texture(names, want, 0)
    return None


def paints_of(names, want):
    """the paint numbers (1, 2 ...) the pack has its own copy of the texture for"""
    w = _stem(want or "")
    return sorted(int(m.group(1)) for m in (PAINT.match(n) for n in names) if m and _stem(m.group(2)) == w)


def blend_sticker(paint, sticker, flip=False):
    """paint: h x w x 4 uint8 pack texture; sticker: straight-alpha RGBA PIL image (rows as stored in the .vpk).
    The app's blend: rgb = paint * (1 - a) + sticker * a  -> new h x w x 4 uint8"""
    h, w = paint.shape[:2]
    if flip:
        sticker = sticker.transpose(Image.FLIP_TOP_BOTTOM)
    pm = sticker.convert("RGBa")                       # premultiplied, so the resize does not bleed the colour
    if pm.size != (w, h):
        pm = pm.resize((w, h), Image.BOX if pm.width >= w and pm.height >= h else Image.BICUBIC)
    s = np.asarray(pm.convert("RGBa"), np.float32)     # premultiplied rgb, alpha 0..255
    a = s[..., 3:4] / 255.0
    out = paint.astype(np.float32)
    out[..., :3] = out[..., :3] * (1 - a) + s[..., :3]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------
# drawing
# ------------------------------------------------------------------------------------------------
class Gbuf:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.ss = 1                                     # supersampling factor (compose() shrinks the picture back by it)
        self.T = np.full((H, W), -1, np.int32)         # -1 nothing, -2 untextured, else texture index
        self.U = np.zeros((H, W), np.float32)
        self.V = np.zeros((H, W), np.float32)
        self.K = np.zeros((H, W, 3), np.float32)       # vertex colour (lighting is applied when composing)
        self.N = np.zeros((H, W, 3), np.float32)       # surface normal, camera space (z towards the viewer)
        self.N[..., 2] = 1
        self.glass = []                                 # see-through fragments, back to front
        self.M = np.full((H, W), -1, np.int32)         # index of the batch that drew the pixel (its material)
        self.batches = None


_GEO = {}


def _geometry(tex, chunks, batches):
    """the triangles of the drawn chunks as flat arrays (model space), cached: turning the car re-uses them"""
    key = (id(batches), id(tex), tuple((c[6], c[7]) for c in chunks if c[8] != WPK_SHADOW))
    hit = _GEO.get(key)
    if hit is not None and hit[0] is batches and hit[1] is tex:      # (the entry keeps them alive, ids stay unique)
        return hit[2]
    Ps, UVs, Cs, Ts, Bl, Cut, Ns, Ms = [], [], [], [], [], [], [], []
    for c in chunks:
        if c[8] == WPK_SHADOW:
            continue
        for bi in range(c[6], c[6] + c[7]):
            B = batches[bi]
            V, I = B["V"], B["I"]
            if not len(I):
                continue
            Ps.append(V["pos"].astype(np.float64)[I])
            UVs.append(V["uv"][I].astype(np.float64))
            Cs.append(V["rgba"][I].astype(np.float64) / 255)
            Ts.append(np.full(len(I), B["tex"], np.int32))
            Ns.append(V["normal"].astype(np.float64)[I] / 32767.0)
            Ms.append(np.full(len(I), bi, np.int32))
            Bl.append(np.full(len(I), B["flags"] & 3, np.int8))     # 0 opaque, 1 alpha, 2 additive
            # the game alpha-tests (discards alpha < 0.5) an opaque batch whose texture or vertices have alpha,
            # except the car shader's (prog 1), which writes alpha 1
            ti = B["tex"]
            has_a = (ti < len(tex) and int(tex[ti][1][..., 3].min()) < 255) or int(V["rgba"][:, 3].min()) < 128
            Cut.append(np.full(len(I), bool(has_a and B["prog"] != WPK_LIT_CAR)))
    if Ps:
        out = (np.concatenate(Ps), np.concatenate(UVs), np.concatenate(Cs), np.concatenate(Ts), np.concatenate(Bl),
               np.concatenate(Cut), np.concatenate(Ns), np.concatenate(Ms))
    else:
        out = (np.zeros((0, 3, 3)),) + (None,) * 7
    if len(_GEO) > 6:
        _GEO.clear()
    _GEO[key] = (batches, tex, out)
    return out


SUPERSAMPLE = 2        # full-size views are drawn at this many times the size and shrunk back: smooth edges (1 = off, faster)


def rasterize(model, yaw, pitch, W=640, H=360, step=None, cancelled=lambda: False, frame=None, zoom=1.0, ss=1, _margin=24):
    """one view of the car -> Gbuf (None if cancelled). step(done, total) reports progress.
    frame: chunks the picture is fitted to (default: the ones drawn), so hiding a wing does not zoom the car.
    zoom: 1 = the car fills the picture, below 1 = the camera is further back (the view is orthographic)."""
    if ss > 1:                                          # supersampled: draw big, compose() shrinks it back to W x H
        g = rasterize(model, yaw, pitch, W * ss, H * ss, step, cancelled, frame, zoom, 1, 24 * ss)
        if g is not None:
            g.ss = ss
        return g
    tex, chunks, batches = model
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    R = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Pm, UV, C, TI, BL, CUT, NVm, MB = _geometry(tex, chunks, batches)
    g = Gbuf(W, H)
    g.batches = batches
    if not len(Pm):
        return g
    P = Pm @ R.T
    lo, hi = P.reshape(-1, 3).min(0), P.reshape(-1, 3).max(0)
    if frame:
        fp = [batches[bi]["V"]["pos"].astype(np.float64)[batches[bi]["I"].ravel()] @ R.T
              for c in frame if c[8] != WPK_SHADOW for bi in range(c[6], c[6] + c[7]) if len(batches[bi]["I"])]
        if fp:
            fp = np.concatenate(fp)
            lo, hi = np.minimum(lo, fp.min(0)), np.maximum(hi, fp.max(0))
    s = min((W - _margin) / max(hi[0] - lo[0], 1e-6), (H - _margin) / max(hi[1] - lo[1], 1e-6)) * zoom
    cx, cyy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    SX = (P[..., 0] - cx) * s + W / 2
    SY = H / 2 - (P[..., 1] - cyy) * s
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    nl = np.maximum(np.linalg.norm(n, axis=1), 1e-12)
    nrm = (n / nl[:, None]).astype(np.float32)
    # smooth (per-vertex) normals in camera space: they make the lighting and the reflection flow over the body.
    # A vertex without a normal (zero) takes the triangle's.
    NV = (NVm @ R.T).astype(np.float32)
    NV = np.where((np.abs(NV).sum(-1, keepdims=True) > 1e-3), NV, nrm[:, None, :])

    def smooth(k, w0, w1, w2):
        v = w0[:, None] * NV[k, 0] + w1[:, None] * NV[k, 1] + w2[:, None] * NV[k, 2]
        ln = np.linalg.norm(v, axis=-1, keepdims=True)
        return np.where(ln > 1e-4, v / np.maximum(ln, 1e-4), nrm[k] if np.ndim(k) == 0 else nrm[k])
    shade = 0.45 + 0.55 * np.abs(n[:, 2] / nl)         # the "Soft" lighting; opaque pixels re-light when composed
    x0 = np.maximum(SX.min(1), 0).astype(int)
    x1 = np.minimum(SX.max(1) + 1, W - 1).astype(int)
    y0 = np.maximum(SY.min(1), 0).astype(int)
    y1 = np.minimum(SY.max(1) + 1, H - 1).astype(int)
    keep = (n[:, 2] > 0) & (x1 > x0) & (y1 > y0)
    ids = np.nonzero(keep)[0]
    opaque = ids[BL[ids] == 0]
    glass = ids[BL[ids] != 0]
    glass = glass[np.argsort(P[glass][:, :, 2].mean(1), kind="stable")]
    zb = np.full((H, W), -1e9)
    total = len(opaque) + len(glass)
    done = 0

    def draw(k, write):
        sx, sy_ = SX[k], SY[k]
        xa, xb, ya, yb = x0[k], x1[k], y0[k], y1[k]
        xs, ys = np.meshgrid(np.arange(xa, xb + 1) + 0.5, np.arange(ya, yb + 1) + 0.5)
        d = (sy_[1] - sy_[2]) * (sx[0] - sx[2]) + (sx[2] - sx[1]) * (sy_[0] - sy_[2])
        if abs(d) < 1e-9:
            return
        l0 = ((sy_[1] - sy_[2]) * (xs - sx[2]) + (sx[2] - sx[1]) * (ys - sy_[2])) / d
        l1 = ((sy_[2] - sy_[0]) * (xs - sx[2]) + (sx[0] - sx[2]) * (ys - sy_[2])) / d
        l2 = 1 - l0 - l1
        m = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not m.any():
            return
        z = l0 * P[k, 0, 2] + l1 * P[k, 1, 2] + l2 * P[k, 2, 2]
        sub = zb[ya:yb + 1, xa:xb + 1]
        vis = m & (z > sub)
        if not vis.any():
            return
        l0, l1, l2 = l0[vis], l1[vis], l2[vis]
        u = (l0 * UV[k, 0, 0] + l1 * UV[k, 1, 0] + l2 * UV[k, 2, 0]) % 1
        v = (l0 * UV[k, 0, 1] + l1 * UV[k, 1, 1] + l2 * UV[k, 2, 1]) % 1
        if write and CUT[k]:                           # alpha test: the cut-out texels are not drawn
            ti_ = int(TI[k])
            a_ = l0 * C[k, 0, 3] + l1 * C[k, 1, 3] + l2 * C[k, 2, 3]
            if ti_ < len(tex):
                a_ = a_ * _sample(tex[ti_][1], u, v)[:, 3]
            ok = a_ >= 0.5
            if not ok.any():
                return
            yy, xx = np.nonzero(vis)
            vis = np.zeros_like(vis)
            vis[yy[ok], xx[ok]] = True
            l0, l1, l2, u, v = l0[ok], l1[ok], l2[ok], u[ok], v[ok]
        col = (l0[:, None] * C[k, 0, :3] + l1[:, None] * C[k, 1, :3] + l2[:, None] * C[k, 2, :3])
        ti = int(TI[k])
        if write:
            sub[vis] = z[vis]
            gt = g.T[ya:yb + 1, xa:xb + 1]
            gt[vis] = -2 if ti == 0xFFFF else ti
            g.U[ya:yb + 1, xa:xb + 1][vis] = u
            g.V[ya:yb + 1, xa:xb + 1][vis] = v
            g.K[ya:yb + 1, xa:xb + 1][vis] = col
            g.N[ya:yb + 1, xa:xb + 1][vis] = smooth(k, l0, l1, l2)
            g.M[ya:yb + 1, xa:xb + 1][vis] = MB[k]
        else:
            col = col * shade[k]
            al = l0 * C[k, 0, 3] + l1 * C[k, 1, 3] + l2 * C[k, 2, 3]
            yy, xx = np.nonzero(vis)
            g.glass.append(((yy + ya) * W + (xx + xa), -2 if ti == 0xFFFF else ti,
                            u.astype(np.float32), v.astype(np.float32), col.astype(np.float32), al.astype(np.float32),
                            int(BL[k]), smooth(k, l0, l1, l2).astype(np.float32), int(MB[k])))

    # Small opaque triangles (nearly all of a car) are rasterized in bulk with numpy; big ones, the alpha-tested
    # ones and the see-through ones take the one-by-one path.
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    bulk = opaque[~CUT[opaque]]
    rest = opaque[CUT[opaque]]
    zflat = zb.reshape(-1)
    Tf, Uf, Vf = g.T.reshape(-1), g.U.reshape(-1), g.V.reshape(-1)
    Kf, Nf, Mf = g.K.reshape(-1, 3), g.N.reshape(-1, 3), g.M.reshape(-1)
    done_ids = np.zeros(len(P), bool)
    for cap in (4, 8, 16, 32):
        sel = bulk[(bw[bulk] <= cap) & (bh[bulk] <= cap) & ~done_ids[bulk]]
        done_ids[sel] = True
        step_n = max(1, 1500000 // (cap * cap))
        for a in range(0, len(sel), step_n):
            if cancelled():
                return None
            idx = sel[a:a + step_n]
            ar = np.arange(cap)
            gx = x0[idx][:, None, None] + ar[None, None, :]
            gy = y0[idx][:, None, None] + ar[None, :, None]
            px, py = gx + 0.5, gy + 0.5
            a0, a1, a2 = SX[idx, 0][:, None, None], SX[idx, 1][:, None, None], SX[idx, 2][:, None, None]
            b0, b1, b2 = SY[idx, 0][:, None, None], SY[idx, 1][:, None, None], SY[idx, 2][:, None, None]
            d = (b1 - b2) * (a0 - a2) + (a2 - a1) * (b0 - b2)
            ok_d = np.abs(d) >= 1e-9
            d = np.where(ok_d, d, 1.0)
            l0 = ((b1 - b2) * (px - a2) + (a2 - a1) * (py - b2)) / d
            l1 = ((b2 - b0) * (px - a2) + (a0 - a2) * (py - b2)) / d
            l2 = 1 - l0 - l1
            z = (l0 * P[idx, 0, 2][:, None, None] + l1 * P[idx, 1, 2][:, None, None] + l2 * P[idx, 2, 2][:, None, None])
            pix = gy * W + gx
            inside = ((l0 >= 0) & (l1 >= 0) & (l2 >= 0) & ok_d & (gx <= x1[idx][:, None, None])
                      & (gy <= y1[idx][:, None, None]))
            inside &= z > zflat[np.where(inside, pix, 0)]
            t_i, y_i, x_i = np.nonzero(inside)
            if not len(t_i):
                continue
            fp, fz, ft = pix[t_i, y_i, x_i], z[t_i, y_i, x_i], idx[t_i]
            o = np.lexsort((fz, fp))                       # per pixel the nearest fragment is the last one
            fp_s = fp[o]
            win = o[np.r_[fp_s[1:] != fp_s[:-1], True]]
            wp, wz, wt = fp[win], fz[win], ft[win]
            w0, w1, w2 = l0[t_i, y_i, x_i][win], l1[t_i, y_i, x_i][win], l2[t_i, y_i, x_i][win]
            zflat[wp] = wz
            Tf[wp] = np.where(TI[wt] == 0xFFFF, -2, TI[wt])
            Uf[wp] = (w0 * UV[wt, 0, 0] + w1 * UV[wt, 1, 0] + w2 * UV[wt, 2, 0]) % 1
            Vf[wp] = (w0 * UV[wt, 0, 1] + w1 * UV[wt, 1, 1] + w2 * UV[wt, 2, 1]) % 1
            Kf[wp] = w0[:, None] * C[wt, 0, :3] + w1[:, None] * C[wt, 1, :3] + w2[:, None] * C[wt, 2, :3]
            Nf[wp] = smooth(wt, w0, w1, w2)
            Mf[wp] = MB[wt]
        if step:
            step(int(done_ids.sum()), total)
    rest = np.concatenate([rest, bulk[~done_ids[bulk]]])
    for j, k in enumerate(rest):
        draw(k, True)
        if j % 400 == 0:
            if cancelled():
                return None
            if step:
                step(j, total)
    for j, k in enumerate(glass):
        draw(k, False)
        if j % 400 == 0:
            if cancelled():
                return None
            if step:
                step(len(opaque) + j, total)
    if step:
        step(total, total)
    return g


_OPAQUE = {}


def _is_opaque(im):
    """True when the picture has no alpha channel or all of it is 255 (cached: the pictures stay alive in the model)"""
    if im.ndim < 3 or im.shape[2] < 4:
        return True
    hit = _OPAQUE.get(id(im))
    if hit is not None and hit[0] is im:
        return hit[1]
    if len(_OPAQUE) > 96:
        _OPAQUE.clear()
    op = bool(im[..., 3].min() == 255)
    _OPAQUE[id(im)] = (im, op)
    return op


def _sample(im, u, v):
    """bilinear texture lookup (what the game's GL_LINEAR does) -> n x 4 floats 0..1; transparent texels do not bleed
    their colour into the edges (alpha is weighted in), the picture wraps around like a repeating texture"""
    h, w = im.shape[:2]
    x = np.asarray(u, np.float32) * w - 0.5
    y = np.asarray(v, np.float32) * h - 0.5
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    xa = x0.astype(np.int64) % w
    ya = y0.astype(np.int64) % h
    xb = (xa + 1) % w
    yb = (ya + 1) % h
    if im.ndim == 2:
        im = np.repeat(im[..., None], 3, axis=2)
    if _is_opaque(im):
        def tap(yy, xx):
            return im[yy, xx].astype(np.float32)
        top = tap(ya, xa) * (1 - fx) + tap(ya, xb) * fx
        bot = tap(yb, xa) * (1 - fx) + tap(yb, xb) * fx
        out = (top * (1 - fy) + bot * fy) / 255
        if out.shape[1] < 4:
            out = np.concatenate([out, np.ones((len(out), 4 - out.shape[1]), np.float32)], axis=1)
        else:
            out[:, 3] = 1.0
        return out
    def tap(yy, xx):
        t = im[yy, xx].astype(np.float32)
        t[:, :3] *= t[:, 3:4] / 255                       # premultiplied
        return t
    top = tap(ya, xa) * (1 - fx) + tap(ya, xb) * fx
    bot = tap(yb, xa) * (1 - fx) + tap(yb, xb) * fx
    out = (top * (1 - fy) + bot * fy) / 255
    al = out[:, 3:4]
    out[:, :3] = out[:, :3] / np.maximum(al, 1e-4)
    return out


LIGHTS = ("Flat", "Soft", "Shadow")
_SUN = np.array([-0.45, 0.65, 0.62]) / np.linalg.norm([-0.45, 0.65, 0.62])


BG_NIGHT = np.array([0.08, 0.11, 0.19], np.float32)                           # moonlit blue-grey (was near-black)
_MOON = np.array([0.5, 0.55, 0.67]) / np.linalg.norm([0.5, 0.55, 0.67])      # a cool light from the upper right
_NIGHT_TINT = np.array([0.80, 0.90, 1.15], np.float32)
NIGHT_ENV = 0.60                                                              # the daylight environment map, dimmed


def lighting(g, light="Soft", night=False):
    """H x W x 1 brightness for the opaque pixels: Flat = none, Soft = the old headlight shading,
    Shadow = one sun from the upper left (dark sides, to judge how a sticker looks on the shaded car)"""
    nz = g.N[..., 2]
    if light == "Flat":
        f = np.ones_like(nz)
    elif light == "Shadow":
        f = 0.22 + 0.9 * np.clip(g.N @ _SUN.astype(np.float32), 0, 1)
    else:
        f = 0.45 + 0.55 * np.abs(nz)
    if night:                      # dark body: a little ambient, a cool moon / street light, blue-ish tint
        moon = np.clip(g.N @ _MOON.astype(np.float32), 0, 1)
        f = (0.20 + 0.12 * f + 0.55 * moon)[..., None] * _NIGHT_TINT   # bright moonlight
        return f.astype(np.float32)
    return f[..., None]


REFLECTIONS = {"Off": 0.0, "Subtle": 0.5, "Full": 1.0}
_HALF = np.array([0.0, 0.0, 1.0]) + _SUN
_HALF = (_HALF / np.linalg.norm(_HALF)).astype(np.float32)      # the highlight's half vector (eye = +z)


def _env_paint(env, N, K, power, refl):
    """car.cg's paint: the sphere-mapped environment (view-space normal), scaled by its own brightness,
    weighted 1 - power, plus a sharp highlight (power 256). N: n x 3 camera space, K: n x 3 lit vertex colour."""
    u = N[:, 0] * 0.5 + 0.5
    v = (N[:, 1] * 0.5 + 0.5) * 0.5 + 0.5
    e = _sample(env, u, v)[:, :3] * K
    e = e * (e * e).sum(1, keepdims=True)
    sp = np.clip(N @ _HALF, 0, 1) ** 256
    return refl * ((1.0 - power) * e + sp[:, None])


def _env_blend(env, spec, N, u, v):
    """car_blend.cg (glass, lamps, wheels): the pack's specular map decides how much of the reflection replaces the
    texture. -> (reflection colour n x 3, share n x 1)"""
    r = 2.0 * N[:, 2:3] * N[:, :2]                      # the view ray runs along -z: its reflection's x, y
    e = _sample(env, r[:, 0] * 0.5 + 0.5, (r[:, 1] * 0.5 + 0.5) * 0.4 + 0.6)[:, :3]
    return e, _sample(spec, u, v)[:, :1]


def _material(g, textures, bi):
    """(prog, env picture, spec picture, power) of batch bi, None for a picture the pack lacks"""
    B = g.batches[bi]
    env = textures[B["env"]][1] if B["env"] < len(textures) else None
    spec = textures[B["spec"]][1] if B["spec"] < len(textures) else None
    return B["prog"], env, spec, B["power"]


def lamp_kind(g, textures, bi):
    """'brake' for the brake-lamp parts (car_brakelamp_* textures), else 'lamp' (any other see-through additive part)"""
    B = g.batches[bi]
    nm = textures[B["tex"]][0].lower() if B["tex"] < len(textures) else ""
    return "brake" if "brake" in nm else "lamp"


def compose(g, textures, override=None, light="Soft", reflection=1.0, night=False, brake=True, lamps=True):
    """Gbuf + textures [(name, rgba)] (+ {index: rgba} replacing some) -> PIL image.
    reflection: 0 = off, 1 = the game's environment reflection and highlight (paint) / reflection (glass, wheels)"""
    override = override or {}
    H, W = g.H, g.W
    img = np.empty((H, W, 3), np.float32)
    img[:] = BG_NIGHT if night else BG
    lit = lighting(g, light, night)
    if night:
        reflection = reflection * NIGHT_ENV
    flat = img.reshape(-1, 3)
    Mf, Tf, Uf, Vf = g.M.reshape(-1), g.T.reshape(-1), g.U.reshape(-1), g.V.reshape(-1)
    Kf, Nf, Lf = g.K.reshape(-1, 3), g.N.reshape(-1, 3), lit.reshape(-1, lit.shape[-1])
    have = np.nonzero(Mf >= 0)[0]
    if len(have):
        order = have[np.argsort(Mf[have], kind="stable")]
        ms = Mf[order]
        cut = np.nonzero(np.r_[True, ms[1:] != ms[:-1], True])[0]
        for a, b in zip(cut[:-1], cut[1:]):
            ix, bi = order[a:b], int(ms[a])
            t = int(Tf[ix[0]])
            col = Kf[ix] * Lf[ix]
            im = override.get(t, textures[t][1]) if 0 <= t < len(textures) else None
            if im is None:
                flat[ix] = col
                continue
            tx = _sample(im, Uf[ix], Vf[ix])[:, :3]
            prog, env, spec, power = _material(g, textures, bi) if g.batches else (0, None, None, 0.0)
            if reflection > 0 and prog == 1 and env is not None:
                flat[ix] = col * tx + _env_paint(env, Nf[ix], col, power, reflection)
            elif reflection > 0 and prog == 2 and env is not None and spec is not None:
                e, sh = _env_blend(env, spec, Nf[ix], Uf[ix], Vf[ix])
                sh = sh * reflection
                flat[ix] = col * (tx * (1 - sh) + e * sh)
            else:
                flat[ix] = col * tx
    # pixels with no material record (an old Gbuf): the old way
    rest = np.nonzero((Mf < 0) & (Tf != -1))[0]
    if len(rest):
        flat[rest] = Kf[rest] * Lf[rest]
    glow = np.zeros_like(flat) if night else None
    for idx, t, u, v, col, al, mode, nrm_g, bi in g.glass:
        if mode == 2 and g.batches:
            if not (brake if lamp_kind(g, textures, bi) == "brake" else lamps):
                continue                                # this lamp is switched off
        if night and mode != 2:
            col = col * lit.reshape(-1, lit.shape[-1])[idx] if lit.shape[-1] == 3 else col
        if t == -2 or t >= len(textures):
            px, a = col, al[:, None]
        else:
            im = override.get(int(t), textures[int(t)][1])
            p = _sample(im, u, v)
            rgb = p[:, :3]
            if reflection > 0 and g.batches and mode != 2:
                prog, env, spec, _pw = _material(g, textures, bi)
                if prog == 2 and env is not None and spec is not None:
                    e, sh = _env_blend(env, spec, nrm_g, u, v)
                    sh = sh * reflection
                    rgb = rgb * (1 - sh) + e * sh
            px, a = col * rgb, (al * p[:, 3])[:, None]
        if mode == 2:                                   # additive (lamp glow): src * alpha + dst
            if night:                                   # lit lamps shine in the dark, and bloom around themselves
                px = px * 1.7
                glow[idx] += px * a
            flat[idx] = np.minimum(flat[idx] + px * a, 1.0)
        else:
            flat[idx] = px * a + flat[idx] * (1 - a)
    if night and glow.any():
        from PIL import ImageFilter
        gl = Image.fromarray((np.clip(glow.reshape(H, W, 3), 0, 1) * 255).astype(np.uint8))
        r = max(2.0, W / 90.0)
        bloom = (np.asarray(gl.filter(ImageFilter.GaussianBlur(r)), np.float32) * 1.3 +
                 np.asarray(gl.filter(ImageFilter.GaussianBlur(r * 3)), np.float32) * 1.6) / 255.0
        img = np.minimum(img + bloom, 1.0)
    out = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    ss = getattr(g, "ss", 1)
    if ss > 1:
        out = out.resize((W // ss, H // ss), Image.LANCZOS)
    return out


# ------------------------------------------------------------------------------------------------
# UV template: the exact layout of the car's body texture, to paint stickers on
# ------------------------------------------------------------------------------------------------
def default_chunks(model):
    """body + fixed parts + the first variant (A) of every part group, no wing: what the preview starts with"""
    hide = set()
    for name, variants in PART_GROUPS:
        for i, v in enumerate(variants):
            if name == "Wing" or i > 0:
                hide.update(v)
    return [c for c in model[1] if c[8] not in hide]


def body_stem(model):
    """stem of the car's body texture (what the .vpk calls 'fc3s_body01_white'), guessed from the pack"""
    for n, _ in model[0]:
        if "body01" in n.lower() and not PAINT.match(n):
            return _stem(n)
    return _stem(model[0][0][0]) if model[0] else ""


def _uv_geometry(model, stem, chunks=None):
    """triangles of texture `stem` in UV space -> (T n x 3 x 2 in 0..1, vertex ids n x 3, unique-vertex keys,
    island number per triangle, number of islands, border-edge keys)"""
    tex, allc, bat = model
    names = [n for n, _ in tex]
    tris = []
    for c in (allc if chunks is None else chunks):
        if c[8] == WPK_SHADOW:
            continue
        for bi in range(c[6], c[6] + c[7]):
            B = bat[bi]
            if B["tex"] >= len(names) or _stem(names[B["tex"]]) != _stem(stem) or not len(B["I"]):
                continue
            tris.append(B["V"]["uv"].astype(np.float64)[B["I"]] % 1.0)
    if not tris:
        raise WpkError("No triangle of this car uses the texture '%s'." % stem)
    T = np.concatenate(tris)                                       # n x 3 x 2
    key = np.round(T * 4095).astype(np.int64)
    key = key[..., 0] * 8192 + key[..., 1]                         # one number per UV position
    uniq, ids = np.unique(key.ravel(), return_inverse=True)
    ids = ids.reshape(-1, 3)
    parent = list(range(len(uniq)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b, c in ids:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[rc] = ra
    root = np.array([find(a) for a in ids[:, 0]])
    isl, island_of = np.unique(root, return_inverse=True)
    e = np.concatenate([ids[:, [0, 1]], ids[:, [1, 2]], ids[:, [2, 0]]])
    e.sort(1)
    ek = e[:, 0].astype(np.int64) * len(uniq) + e[:, 1]
    eu, cnt = np.unique(ek, return_counts=True)
    return T, ids, uniq, island_of, len(isl), eu[cnt == 1]


_SEG = {}


def uv_segments(model, stem, chunks=None):
    """the real UV borders as line segments: array n x 4 (x0, y0, x1, y1) in 0..1, y down like a sticker PNG.
    Vector data, so they can be drawn crisp at any size or zoom."""
    key = (id(model[2]), _stem(stem), tuple(c[8] for c in (model[1] if chunks is None else chunks)))
    hit = _SEG.get(key)
    if hit is not None and hit[0] is model[2]:
        return hit[1]
    _T, _ids, uniq, _isl, _n, border = _uv_geometry(model, stem, chunks)
    a, b = border // len(uniq), border % len(uniq)
    u = (uniq // 8192) / 4095.0
    v = (uniq % 8192) / 4095.0
    seg = np.stack([u[a], 1 - v[a], u[b], 1 - v[b]], 1)
    if len(_SEG) > 12:
        _SEG.clear()
    _SEG[key] = (model[2], seg)
    return seg


def draw_segments(seg, size, view=(0.0, 0.0, 1.0, 1.0), width=2, color=(255, 0, 90, 255)):
    """segments (see uv_segments) -> transparent RGBA picture size x size showing the window view = (x, y, w, h) of the
    sticker (all of it by default). Lines keep `width` screen pixels at any zoom."""
    from PIL import ImageDraw
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x, y, w_, h_ = view
    sx, sy = size / w_, size / h_
    X0, Y0 = (seg[:, 0] - x) * sx, (seg[:, 1] - y) * sy
    X1, Y1 = (seg[:, 2] - x) * sx, (seg[:, 3] - y) * sy
    vis = ((np.maximum(X0, X1) >= -2) & (np.minimum(X0, X1) <= size + 2)
           & (np.maximum(Y0, Y1) >= -2) & (np.minimum(Y0, Y1) <= size + 2))
    for a, b, c, e in zip(X0[vis], Y0[vis], X1[vis], Y1[vis]):
        d.line((a, b, c, e), fill=color, width=width)
    return img


def uv_template(model, stem, size=2048, chunks=None, labels=True, fill=True, line=(20, 20, 20, 255), png=True, lw=None):
    """Draw the UV layout of texture `stem` (exactly the triangles the game maps it onto) -> (RGBA PIL image, n islands).
    Every island (connected piece of the model) gets its own colour and number; the outline is the real UV border.
    The picture has the same orientation as a sticker PNG (Export PNG / Replace image), so art painted on it
    lines up 1:1 on the car."""
    from PIL import ImageDraw, ImageFont
    import colorsys
    T, ids, uniq, island_of, n_isl, border_keys = _uv_geometry(model, stem, chunks)
    border = set(border_keys.tolist())
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pal = [tuple(int(255 * x) for x in colorsys.hsv_to_rgb((i * 0.381966) % 1.0, 0.45, 0.95)) for i in range(n_isl)]
    P = T * size
    if png:
        P[..., 1] = size - P[..., 1]        # sticker PNGs are top-down, the texture's V runs bottom-up
    for k in range(len(T)) if fill else ():
        d.polygon([tuple(p) for p in P[k]], fill=pal[island_of[k]] + (255,))
    lw = lw or (max(1, size // 1024) if fill else max(2, size // 384))
    for k in range(len(T)):
        v = ids[k]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            lo, hi = sorted((v[a], v[b]))
            if lo * len(uniq) + hi in border:
                d.line([tuple(P[k][a]), tuple(P[k][b])], fill=line, width=lw)
    if labels:
        try:
            font = ImageFont.load_default(size=max(10, size // 64))
        except TypeError:
            font = ImageFont.load_default()
        for j in range(n_isl):
            pts = P[island_of == j].reshape(-1, 2)
            w_, h_ = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
            if w_ * h_ > (size * 0.02) ** 2:
                cx, cy = pts.mean(0)
                d.text((cx, cy), str(j + 1), fill=(0, 0, 0, 255), font=font, anchor="mm")
    return img, n_isl


def export_textures(model, out_dir, only=None):
    """every texture of the car pack as a PNG (the body in each paint, the lamps / plate strip, tyres, wheels, spec
    and environment maps ...) -> list of file names. Rows are as stored in the pack (the lamp strip is at the top)."""
    os.makedirs(out_dir, exist_ok=True)
    done, used = [], set()
    for i, (name, im) in enumerate(model[0]):
        if only is not None and i not in only:
            continue
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(name)[0].replace(":", "_")) or "tex"
        fn = "%02d_%s.png" % (i, stem)
        if fn in used:
            continue
        used.add(fn)
        Image.fromarray(im).save(os.path.join(out_dir, fn))
        done.append(fn)
    return done


def export_all_textures(folder, out_dir, progress=None, cancelled=lambda: False):
    """every texture of every car .wpk under `folder` as PNGs -> (cars done, [(file, why)]).
    One sub-folder per car pack in out_dir; an aero kit gets only the textures it has of its own (the rest are the
    stock car's). progress(done, total, name) is called after each car."""
    import glob
    files = sorted(glob.glob(os.path.join(folder, "**", "*.wpk"), recursive=True))
    os.makedirs(out_dir, exist_ok=True)
    done, failed, rows = 0, [], []
    for n, p in enumerate(files):
        if cancelled():
            break
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            sp = stock_of(p)
            st = read_wpk3(sp)[0] if sp else None
            m = read_wpk3(p, st)
            own = None
            if st:                                              # a borrowed texture is the very same array as the stock's
                borrowed = {id(im) for _n, im in st}
                own = {i for i, (_n, im) in enumerate(m[0]) if id(im) not in borrowed}
            files_out = export_textures(m, os.path.join(out_dir, name), own)
            if own is not None and not files_out:
                os.rmdir(os.path.join(out_dir, name))
                rows.append("%s\tno textures of its own (uses the stock car's)" % name)
            else:
                rows.append("%s\t%d textures" % (name, len(files_out)))
            done += 1
        except Exception as ex:                                 # one broken file must not stop the rest
            failed.append((os.path.basename(p), str(ex)))
        if progress:
            progress(n + 1, len(files), name)
    with open(os.path.join(out_dir, "index.txt"), "w", encoding="utf-8") as f:
        f.write("Car textures as PNG, one folder per car pack, rows as stored in the pack.\n\n")
        f.write("\n".join(rows))
        if failed:
            f.write("\n\nNot exported:\n" + "\n".join("%s\t%s" % x for x in failed))
        f.write("\n")
    return done, failed


def export_all_uv(folder, out_dir, size=2048, progress=None, cancelled=lambda: False):
    """UV template PNG of every car .wpk found under `folder` (stock and aero kits) -> (done, [(file, why)]).
    Written to out_dir as <car>_uv.png with an index.txt. progress(done, total, name) is called after each car."""
    import glob
    files = sorted(glob.glob(os.path.join(folder, "**", "*.wpk"), recursive=True))
    os.makedirs(out_dir, exist_ok=True)
    done, failed, rows = 0, [], []
    for n, p in enumerate(files):
        if cancelled():
            break
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            sp = stock_of(p)
            m = read_wpk3(p, read_wpk3(sp)[0] if sp else None)
            stem = body_stem(m)
            im, islands = uv_template(m, stem, size, default_chunks(m))
            im.save(os.path.join(out_dir, name + "_uv.png"))
            rows.append("%s_uv.png\t%d islands\ttexture %s" % (name, islands, stem))
            done += 1
        except Exception as ex:                     # one broken file must not stop the rest
            failed.append((os.path.basename(p), str(ex)))
        if progress:
            progress(n + 1, len(files), name)
    with open(os.path.join(out_dir, "index.txt"), "w", encoding="utf-8") as f:
        f.write("UV templates, %d x %d px, drawn like the sticker PNGs (Export PNG / Replace image).\n" % (size, size))
        f.write("Each island has its own colour and number.\n\n")
        f.write("\n".join(rows))
        if failed:
            f.write("\n\nNot made:\n" + "\n".join("%s\t%s" % x for x in failed))
        f.write("\n")
    return done, failed


def body_wpk(path, body):
    """the .wpk of body 0 (normal) or 1..3 (aero A..C) of the car whose stock or aero .wpk is `path`, or None"""
    m = KIT_RE.match(os.path.basename(path))
    if not m:
        return path if body == 0 else None
    p = os.path.join(os.path.dirname(os.path.abspath(path)), m.group(1) + ("", "_aeroA", "_aeroB", "_aeroC")[body] + ".wpk")
    return p if os.path.isfile(p) else None


def uv_check(template, sticker):
    """sticker (RGBA, any size) drawn on the template's borders -> (picture, share of the car surface the sticker covers,
    share of the sticker's visible pixels that lie outside every UV island)"""
    size = template.width
    st = sticker.convert("RGBA").resize((size, size), Image.BICUBIC)
    a = np.asarray(st)[..., 3] > 16
    used = np.asarray(template)[..., 3] > 0
    shown = a if not a.all() else (np.asarray(st)[..., :3].sum(-1) > 40)    # fully opaque sheet: black = unused
    inside = (shown & used).sum() / max(used.sum(), 1)
    outside = (shown & ~used).sum() / max(shown.sum(), 1)
    bg = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    bg.alpha_composite(st)
    lines = np.asarray(template).copy()
    border = (lines[..., :3].sum(-1) < 90) & (lines[..., 3] > 0)
    over = np.zeros_like(lines)
    over[border] = (255, 0, 80, 255)
    bg.alpha_composite(Image.fromarray(over))
    return bg.convert("RGB"), inside, outside


# ------------------------------------------------------------------------------------------------
# the window
# ------------------------------------------------------------------------------------------------
def find_wpk(vpk_path, extra_dirs=()):
    """the car's .wpk for a .vpk: next to it, or in a cars folder around it"""
    name = os.path.splitext(os.path.basename(vpk_path))[0] + ".wpk"
    d = os.path.dirname(os.path.abspath(vpk_path))
    cands = list(extra_dirs) + [d, os.path.join(d, "cars"), os.path.join(d, "..", "cars"),
                                os.path.join(d, "..", "..", "cars"), os.path.join(d, "..", "data", "cars"),
                                os.path.join(d, "..", "..", "data", "cars")]
    for c in cands:
        p = os.path.normpath(os.path.join(c, name))
        if os.path.isfile(p):
            return p
    return None


class PreviewWindow:
    """A window showing the car with the current sticker on it. Drive it with set_car / set_sticker."""

    W, H = 640, 360

    def __init__(self, root, ask_wpk, on_close=None):
        import tkinter as tk
        from tkinter import ttk
        from PIL import ImageTk
        self._tk, self._ImageTk = tk, ImageTk
        self.root, self.ask_wpk, self.on_close = root, ask_wpk, on_close
        self.top = tk.Toplevel(root)
        self.top.title("Car preview")
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self.top.resizable(True, True)
        self.top.minsize(720, 720)
        self._rjob = None
        self._full = False
        self.model = None
        self.path = None
        self.gbufs = {}
        self.gen = 0
        self.q = queue.Queue()
        self.sticker = self.original = None
        self.tex_want = ""
        self.photo = None
        self._job = None
        self.v_view = tk.IntVar(value=0)
        self.v_mode = tk.StringVar(value="edited")
        self.v_flip = tk.BooleanVar(value=False)
        self.v_tex = tk.StringVar()
        self.v_paint = tk.StringVar(value="Base paint")
        self.msg = tk.StringVar(value="")
        self.v_kit = tk.StringVar(value="None")
        self.v_zoom = tk.DoubleVar(value=0.85)        # camera distance: 1 = car fills the picture
        self.free = [-35.0, 14.0]                      # yaw, pitch of the "Free" view (drag the picture to turn the car)
        self.v_light = tk.StringVar(value="Soft")
        self.v_refl = tk.StringVar(value="Full")       # the game's reflection + highlight: Off / Subtle / Full
        self.v_night = tk.BooleanVar(value=False)      # night: dark scene, lamps shine and bloom
        self.v_brake = tk.BooleanVar(value=True)       # brake lamps shown
        self.v_lamps = tk.BooleanVar(value=True)       # the other lamps (head, fog, turn ...) shown
        self._pjob = None
        pr = load_prefs().get("preview", {})
        for var, key in ((self.v_light, "light"), (self.v_refl, "reflection"), (self.v_night, "night"),
                         (self.v_brake, "brake"), (self.v_lamps, "lamps")):
            if key in pr:
                try:
                    var.set(pr[key])
                except (self._tk.TclError, ValueError):
                    pass
        if self.v_light.get() not in LIGHTS:
            self.v_light.set("Soft")
        if self.v_refl.get() not in REFLECTIONS:
            self.v_refl.set("Full")
        try:
            gw, gh = int(pr.get("width", 0)), int(pr.get("height", 0))
            if 720 <= gw <= self.top.winfo_screenwidth() and 720 <= gh <= self.top.winfo_screenheight():
                self.top.geometry("%dx%d" % (gw, gh))
        except (TypeError, ValueError, self._tk.TclError):
            pass
        for var in (self.v_light, self.v_refl, self.v_night, self.v_brake, self.v_lamps):
            var.trace_add("write", lambda *_a: self.remember())
        self.v_uvcar = tk.StringVar(value="Off")       # UV layout drawn on the car's texture: Off / Borders / Checker
        self.v_uvop = tk.DoubleVar(value=0.8)
        self._drag = None
        self._lr_busy = self._lr_dirty = False
        self._zjob = None
        self._uvc = {}
        self._pending = set()                          # views being drawn
        self._sver, self._ovr_key, self._ovr = 0, None, {}
        self._lr_div = 2                               # drag frames are 1/div of the picture; adapts to the machine
        self._slider_ok = True
        self.photo = None
        self._lrc = {}                                 # recent drag frames, by angle
        self.kit_files = {}
        self.v_part = {n: tk.StringVar(value="") for n, _v in PART_GROUPS}     # "" = not chosen yet
        self.cb_part = {}

        fr = ttk.Frame(self.top, padding=(10, 8, 10, 8))
        fr.pack(fill="both", expand=True)
        fr.columnconfigure(0, weight=1)
        fr.rowconfigure(1, weight=1)                       # the picture takes whatever room the panels leave
        PAD = dict(padx=(0, 8), pady=2)

        # ---- top: which view, which car file, maximize
        bar = ttk.Frame(fr)
        bar.grid(row=0, column=0, sticky="ew")
        for i, (nm, _y, _p) in enumerate(VIEWS):
            ttk.Radiobutton(bar, text=nm, value=i, variable=self.v_view, command=self.show,
                            style="Toolbutton").pack(side="left", padx=1)
        ttk.Radiobutton(bar, text="Free", value=FREE, variable=self.v_view, command=self.show,
                        style="Toolbutton").pack(side="left", padx=1)
        ttk.Button(bar, text="Maximize", width=11, command=self.toggle_max).pack(side="right")
        ttk.Button(bar, text="Car file…", width=11, command=self.pick).pack(side="right", padx=(0, 6))

        # ---- the picture
        self.canvas = tk.Canvas(fr, width=self.W, height=self.H, highlightthickness=1, highlightbackground="#999",
                                background="#8c9eb3")
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=8)
        self.canvas.bind("<Configure>", self.canvas_resized)
        self.canvas.bind("<Double-Button-1>", lambda _e: self.toggle_max())
        self.top.bind("<F11>", lambda _e: self.toggle_max())
        self.canvas.bind("<ButtonPress-1>", self.drag_start)
        self.canvas.bind("<B1-Motion>", self.drag_move)
        self.canvas.bind("<ButtonRelease-1>", self.drag_end)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind(seq, self.wheel_zoom)

        # ---- three panels side by side: Car (what is drawn) | Scene (how it is lit) | Sticker (what is on it)
        panels = ttk.Frame(fr)
        panels.grid(row=2, column=0, sticky="ew")
        for c in range(3):
            panels.columnconfigure(c, weight=1, uniform="panel")

        def panel(col, title):
            f = ttk.LabelFrame(panels, text=title, padding=(8, 4, 4, 6))
            f.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 6, 0))
            f.columnconfigure(1, weight=1)
            return f

        def line(f, r, text, widget):
            ttk.Label(f, text=text).grid(row=r, column=0, sticky="w", **PAD)
            widget.grid(row=r, column=1, sticky="ew", pady=2)
            return widget

        car = panel(0, "Car")
        self.cb_paint = line(car, 0, "Paint", ttk.Combobox(car, textvariable=self.v_paint, state="readonly",
                                                           width=14, values=["Base paint"]))
        self.cb_paint.bind("<<ComboboxSelected>>", lambda _e: (self.pick_texture(), self.show()))
        self.cb_tex = line(car, 1, "Texture", ttk.Combobox(car, textvariable=self.v_tex, state="readonly", width=14))
        self.cb_tex.bind("<<ComboboxSelected>>", lambda _e: self.show())
        self.cb_kit = line(car, 2, "Aero kit", ttk.Combobox(car, textvariable=self.v_kit, state="readonly",
                                                            width=14, values=["None"]))
        self.cb_kit.bind("<<ComboboxSelected>>", lambda _e: self.pick_kit())
        for r, (name, _variants) in enumerate(PART_GROUPS, start=3):
            cb = line(car, r, name, ttk.Combobox(car, textvariable=self.v_part[name], state="disabled", width=14,
                                                 values=["None"]))
            cb.bind("<<ComboboxSelected>>", lambda _e: self.parts_changed())
            self.cb_part[name] = cb

        scene = panel(1, "Scene")
        cbl = line(scene, 0, "Lighting", ttk.Combobox(scene, textvariable=self.v_light, state="readonly", width=14,
                                                      values=list(LIGHTS)))
        cbl.bind("<<ComboboxSelected>>", lambda _e: self.show())
        cbr = line(scene, 1, "Reflection", ttk.Combobox(scene, textvariable=self.v_refl, state="readonly", width=14,
                                                        values=list(REFLECTIONS)))
        cbr.bind("<<ComboboxSelected>>", lambda _e: self.show())
        checks = ttk.Frame(scene)
        line(scene, 2, "Night", checks)
        ttk.Checkbutton(checks, text="On", variable=self.v_night, command=self.show).pack(side="left")
        ttk.Checkbutton(checks, text="Brake lamps", variable=self.v_brake, command=self.show).pack(side="left", padx=8)
        ttk.Checkbutton(checks, text="Other lamps", variable=self.v_lamps, command=self.show).pack(side="left")
        cam = ttk.Frame(scene)
        line(scene, 3, "Camera", cam)
        sc = ttk.Scale(cam, from_=2.0, to=0.3, variable=self.v_zoom, command=lambda _v: self.slider_moved())
        sc.pack(side="left", fill="x", expand=True)
        sc.bind("<ButtonRelease-1>", lambda _e: self.zoom_to(float(self.v_zoom.get()), True))
        self.zoom_lab = ttk.Label(cam, text="85%", width=5, anchor="e")
        self.zoom_lab.pack(side="left", padx=(6, 0))
        ttk.Button(cam, text="Reset", width=6, command=self.reset_zoom).pack(side="left", padx=(6, 0))
        ttk.Label(scene, text="wheel over the picture = camera distance, drag = turn the car",
                  foreground="#777").grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))

        st = panel(2, "Sticker and UV")
        self._modes = {"With your changes": "edited", "Original sticker": "original", "No sticker": "none"}
        self.v_modetxt = tk.StringVar(value="With your changes")
        cbm = line(st, 0, "Show", ttk.Combobox(st, textvariable=self.v_modetxt, state="readonly", width=14,
                                                values=list(self._modes)))
        cbm.bind("<<ComboboxSelected>>", lambda _e: (self.v_mode.set(self._modes.get(self.v_modetxt.get(), "edited")),
                                                     self.show()))
        fl = ttk.Checkbutton(st, text="Flip upside down", variable=self.v_flip, command=self.show)
        line(st, 1, "Orientation", fl)
        cbu = line(st, 2, "UV on the car", ttk.Combobox(st, textvariable=self.v_uvcar, state="readonly", width=14,
                                                        values=["Off", "Borders", "Checker"]))
        cbu.bind("<<ComboboxSelected>>", lambda _e: self.show())
        su = line(st, 3, "UV strength", ttk.Scale(st, from_=0.1, to=1.0, variable=self.v_uvop))
        su.bind("<ButtonRelease-1>", lambda _e: self.show())

        # ---- actions: one row of equal buttons
        acts = ttk.Frame(fr)
        acts.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        for c in range(3):
            acts.columnconfigure(c, weight=1, uniform="act")
        for c, (txt, fn) in enumerate((("Screenshot…", self.screenshot), ("Export car textures…", self.export_tex),
                                       ("Export UV template…", self.save_uv))):
            ttk.Button(acts, text=txt, command=fn).grid(row=0, column=c, sticky="ew", padx=(0 if c == 0 else 6, 0))

        # ---- status
        self.prog = ttk.Progressbar(fr, maximum=1.0)
        self.prog.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(fr, textvariable=self.msg, foreground="#555", wraplength=680, justify="left").grid(
            row=5, column=0, sticky="w")
        self.top.after(60, self.pump)

    # ---- window size: the picture follows the window (drag its edge, Maximize, F11 or double-click the picture)
    def toggle_max(self):
        self._full = not self._full
        try:
            self.top.state("zoomed" if self._full else "normal")          # Windows / macOS
        except self._tk.TclError:
            try:
                self.top.attributes("-zoomed", self._full)                # Linux
            except self._tk.TclError:
                pass

    def canvas_resized(self, e):
        w, h = (e.width - 2) // 2 * 2, (e.height - 2) // 2 * 2
        if w < 320 or h < 180 or (abs(w - self.W) < 12 and abs(h - self.H) < 12):
            return
        if self._rjob:
            self.top.after_cancel(self._rjob)
        self._rjob = self.top.after(300, lambda: self._apply_size(w, h))     # redraw once the resizing pauses

    def _apply_size(self, w, h):
        self._rjob = None
        if not self.alive() or (w, h) == (self.W, self.H):
            return
        self.W, self.H = w, h
        self.remember()
        self._lrc = {}
        self._lr_div = 2 if w <= 800 else 3
        if self.model:
            self.start_render()
            self.show()

    # ---- car
    def close(self):
        try:
            self._remember_now()
        except Exception:
            pass
        self.gen += 1
        self.top.destroy()
        if self.on_close:
            self.on_close()

    def alive(self):
        try:
            return bool(self.top.winfo_exists())
        except self._tk.TclError:
            return False

    def pick(self):
        p = self.ask_wpk()
        if p:
            self.set_car(p)

    def set_car(self, path):
        """load a .wpk (the stock car or one of its aero kits); its views are drawn once it is in"""
        if not path or path == self.path:
            return
        self.path = path
        self.model = None
        self.gbufs = {}
        self.gen += 1
        gen = self.gen
        self.refresh_kits()
        self.prog["value"] = 0
        self.msg.set("Loading %s …" % os.path.basename(path))
        self.canvas.delete("all")
        self.photo = None
        self._pending = set()

        def work():
            try:
                sp = stock_of(path)
                model = read_wpk3(path, [(n, a) for n, a in read_wpk3(sp)[0]] if sp else None)
            except WpkError as ex:
                self.q.put((gen, "err", str(ex)))
                return
            except Exception as ex:                       # a damaged file must not kill the window
                self.q.put((gen, "err", "Cannot read the car: %s" % ex))
                return
            self.q.put((gen, "model", model))
        threading.Thread(target=work, daemon=True).start()

    # ---- aero kit (the car's _aeroA / _aeroB / _aeroC .wpk) and optional parts (wing, bonnet ...)
    def refresh_kits(self):
        m = KIT_RE.match(os.path.basename(self.path or ""))
        self.kit_files = {}
        if m:
            d = os.path.dirname(os.path.abspath(self.path))
            for k, suffix in (("None", ""), ("A", "_aeroA"), ("B", "_aeroB"), ("C", "_aeroC")):
                p = os.path.join(d, m.group(1) + suffix + ".wpk")
                if os.path.isfile(p):
                    self.kit_files[k] = p
            self.v_kit.set(m.group(2).upper() if m.group(2) else "None")
        self.cb_kit.configure(values=list(self.kit_files) or [self.v_kit.get()],
                              state="readonly" if len(self.kit_files) > 1 else "disabled")

    def pick_kit(self):
        p = self.kit_files.get(self.v_kit.get())
        if p:
            self.set_car(p)

    def build_parts(self):
        """fill the Wing / Bonnet / Mirrors / Wheels boxes with what this car has; keep earlier choices"""
        tags = {c[8] for c in self.model[1]}
        for name, variants in PART_GROUPS:
            opts = ["None"] + [chr(65 + i) for i, v in enumerate(variants) if all(t in tags for t in v)]
            var = self.v_part[name]
            if var.get() not in opts:
                var.set("None" if name == "Wing" or len(opts) == 1 else opts[1])    # no wing until you pick one
            self.cb_part[name].configure(values=opts, state="readonly" if len(opts) > 1 else "disabled")

    def visible_chunks(self):
        hide = set()
        for name, variants in PART_GROUPS:
            for i, v in enumerate(variants):
                if self.v_part[name].get() != chr(65 + i):
                    hide.update(v)
        return [c for c in self.model[1] if c[8] not in hide]

    def save_uv(self):
        """the exact UV layout of the texture the sticker is painted on, as a PNG to paint on (2048 x 2048)"""
        if not self.model:
            return
        from tkinter import filedialog
        name = PAINT.sub(r"\2", self.v_tex.get() or "")
        stem = _stem(name)
        base = os.path.splitext(os.path.basename(self.path or "car"))[0]
        f = filedialog.asksaveasfilename(parent=self.top, title="Export UV template", defaultextension=".png",
                                         initialfile="%s_uv.png" % base, filetypes=[("PNG", "*.png")])
        if not f:
            return
        try:
            im, n = uv_template(self.model, stem, 2048, self.visible_chunks())
            im.save(f)
            self.msg.set("Saved %s (%d islands, drawn for the parts you picked). Paint on it, export, then Replace image."
                         % (os.path.basename(f), n))
        except Exception as ex:
            self.msg.set("Cannot make the template: %s" % ex)

    # ---- turn the car by dragging, zoom with the wheel / slider
    # Every change shows a quick small frame at once (re-used if seen before), then the full picture replaces it.
    # The old picture stays on screen until a new one is ready, so nothing blanks.
    def invalidate(self):
        """throw away the drawn views (zoom / parts changed); work in progress for them is cancelled"""
        self.gen += 1
        self.gbufs = {}
        self._pending = set()
        self.prog["value"] = 0

    def drag_start(self, e):
        if not self.model:
            return
        self._drag = (e.x, e.y)
        self.v_view.set(FREE)
        self.gbufs.pop(FREE, None)

    def drag_move(self, e):
        if self._drag is None:
            return
        dx, dy = e.x - self._drag[0], e.y - self._drag[1]
        self._drag = (e.x, e.y)
        self.free[0] = (self.free[0] + dx * 0.6) % 360
        self.free[1] = max(-89.0, min(89.0, self.free[1] + dy * 0.4))
        self.gbufs.pop(FREE, None)
        self._pending.discard(FREE)
        self.lowres(FREE)

    def drag_end(self, _e):
        if self._drag is None:
            return
        self._drag = None
        self._pending.add(FREE)
        self.render_free()

    def lowres(self, vi=None):
        """a small, quick frame of view vi (default: the one shown), one at a time, always for the newest settings"""
        if not self.model:
            return
        vi = self.v_view.get() if vi is None else vi
        if self._lr_busy:
            self._lr_dirty = True
            return
        self._lr_busy, self._lr_dirty = True, False
        gen, zoom, div = self.gen, float(self.v_zoom.get()), self._lr_div
        yaw, pit = (round(self.free[0]), round(self.free[1])) if vi == FREE else VIEWS[vi][1:]
        chunks = self.visible_chunks()
        key = (id(self.model), vi, yaw, pit, round(zoom, 3), div, tuple(c[8] for c in chunks))
        hit = self._lrc.get(key)
        if hit is not None:
            self.q.put((gen, "lowres", (vi, hit, 0.0)))
            return
        shown = (self.model[0], chunks, self.model[2])
        allc = self.model[1]

        def work():
            t0, gb = time.time(), None
            try:
                gb = rasterize(shown, yaw, pit, max(self.W // div, 64), max(self.H // div, 36), frame=allc, zoom=zoom)
                if len(self._lrc) >= 60:
                    self._lrc.pop(next(iter(self._lrc)), None)
                self._lrc[key] = gb
            except Exception:
                gb = None
            self.q.put((gen, "lowres", (vi, gb, time.time() - t0)))
        threading.Thread(target=work, daemon=True).start()

    def render_free(self):
        """full-size free view at the angle the car was left at"""
        gen, yaw, pit, zoom = self.gen, self.free[0], self.free[1], float(self.v_zoom.get())
        shown = (self.model[0], self.visible_chunks(), self.model[2])
        allc = self.model[1]

        def work():
            try:
                gb = rasterize(shown, yaw, pit, self.W, self.H, frame=allc, zoom=zoom, ss=SUPERSAMPLE)
            except Exception as ex:
                self.q.put((gen, "err", "Cannot draw the car: %s" % ex))
                return
            self.q.put((gen, "free", (gb, yaw, pit)))
        threading.Thread(target=work, daemon=True).start()

    def slider_moved(self):
        if self.model and self._slider_ok:
            self.zoom_to(float(self.v_zoom.get()), False)
        else:
            self.zoom_lab.configure(text="%d%%" % round(float(self.v_zoom.get()) * 100))

    def zoom_to(self, z, final):
        """new camera distance: a quick frame now; the full picture when final (wheel pause / slider released)"""
        if not self.model:
            return
        self._slider_ok = False
        self.v_zoom.set(z)
        self._slider_ok = True
        self.zoom_lab.configure(text="%d%%" % round(z * 100))
        self.invalidate()
        self.lowres()
        if self._zjob:
            self.top.after_cancel(self._zjob)
            self._zjob = None
        if final:
            self.render_view(self.v_view.get())
        else:
            self._zjob = self.top.after(180, self._zoomed)

    def _zoomed(self):
        self._zjob = None
        self.render_view(self.v_view.get())

    def wheel_zoom(self, e):
        up = e.num == 4 or getattr(e, "delta", 0) > 0
        self.zoom_to(max(0.3, min(2.0, float(self.v_zoom.get()) * (1.1 if up else 1 / 1.1))), False)

    def reset_zoom(self):
        self.zoom_to(0.85, True)

    def parts_changed(self):
        self.start_render()
        self.show()

    def start_render(self):
        """draw the view you are looking at with the chosen parts; the other views are drawn when you switch to them"""
        if not self.model:
            return
        self.invalidate()
        self.lowres()
        self.render_view(self.v_view.get())

    def render_view(self, vi):
        if not self.model or vi in self._pending or vi in self.gbufs:
            return
        if vi == FREE:
            self._pending.add(vi)
            self.render_free()
            return
        self._pending.add(vi)
        gen, zoom = self.gen, float(self.v_zoom.get())
        shown = (self.model[0], self.visible_chunks(), self.model[2])
        allc = self.model[1]
        yaw, pit = VIEWS[vi][1:]

        def work():
            try:
                gb = rasterize(shown, yaw, pit, self.W, self.H, frame=allc, zoom=zoom, ss=SUPERSAMPLE,
                               step=lambda d, t: self.q.put((gen, "prog", d / max(t, 1))),
                               cancelled=lambda: gen != self.gen)
            except Exception as ex:
                self.q.put((gen, "err", "Cannot draw the car: %s" % ex))
                return
            if gb is None:
                return
            self.q.put((gen, "view", (vi, gb)))
            self.q.put((gen, "done", None))
        threading.Thread(target=work, daemon=True).start()

    def pump(self):
        if not self.alive():
            return
        redraw = False
        try:
            while True:
                gen, kind, x = self.q.get_nowait()
                if kind == "lowres":
                    self._lr_busy = False
                    vi, gb, took = x
                    if gb is not None and gen == self.gen and vi == self.v_view.get() and vi not in self.gbufs:
                        self.paint(gb)
                    if took:                                   # slow machine: smaller quick frames; fast: bigger
                        if took > 0.09 and self._lr_div < 4:
                            self._lr_div += 1
                        elif took < 0.03 and self._lr_div > 2:
                            self._lr_div -= 1
                    if self._lr_dirty:
                        self._lr_dirty = False
                        self.lowres()
                    continue
                if gen != self.gen:
                    continue
                if kind == "model":
                    self.model = x
                    names = [n for n, _ in x[0]]
                    self.cb_tex.configure(values=names)
                    self.pick_texture()
                    self.build_parts()
                    self.msg.set("Drawing the views …")
                    self.start_render()
                elif kind == "view":
                    self.gbufs[x[0]] = x[1]
                    self._pending.discard(x[0])
                    redraw = redraw or x[0] == self.v_view.get()
                elif kind == "free":
                    self._pending.discard(FREE)
                    if (x[1], x[2]) == (self.free[0], self.free[1]):
                        self.gbufs[FREE] = x[0]
                        redraw = redraw or self.v_view.get() == FREE
                elif kind == "prog":
                    self.prog["value"] = x
                elif kind == "done":
                    self.prog["value"] = 1
                    self.msg.set("")
                    self.status_text()
                elif kind == "err":
                    self.msg.set(x)
        except queue.Empty:
            pass
        if redraw:
            self.show()
        self.top.after(60, self.pump)

    # ---- sticker
    def set_sticker(self, edited, original, tex_name):
        """edited / original: straight-alpha RGBA PIL images as stored in the .vpk (or None); tex_name: the
        texture of the car pack the sticker is blended onto"""
        self.sticker, self.original = edited, original
        self._sver += 1
        if tex_name != self.tex_want:
            self.tex_want = tex_name
            self.pick_texture()
        if self._job is None:                              # many changes in a row draw once
            self._job = self.top.after(30, self._later)

    def _later(self):
        self._job = None
        self.show()

    def paint_no(self):
        s = self.v_paint.get()
        return int(s.split()[-1]) if s.startswith("Paint ") else 0

    def pick_texture(self):
        """the texture the sticker goes on: the .vpk's, in the chosen paint's copy"""
        if not self.model:
            return
        names = [n for n, _ in self.model[0]]
        ps = paints_of(names, self.tex_want)
        self.cb_paint.configure(values=["Base paint"] + ["Paint %d" % p for p in ps])
        if self.v_paint.get() not in ["Base paint"] + ["Paint %d" % p for p in ps]:
            self.v_paint.set("Base paint")
        i = find_texture(names, self.tex_want, self.paint_no())
        if i is None and not self.v_tex.get() in names:
            i = next((k for k, n in enumerate(names) if "body" in n.lower()), 0 if names else None)
        if i is not None:
            self.v_tex.set(names[i])

    def status_text(self):
        if not self.model:
            return
        names = [n for n, _ in self.model[0]]
        if self.tex_want and find_texture(names, self.tex_want, self.paint_no()) is None:
            self.msg.set("The car has no texture called '%s' (the sticker file's name for it); showing it on '%s'. "
                         "Pick the right texture above if that is wrong." % (self.tex_want, self.v_tex.get()))

    # ---- picture
    def textures_override(self):
        """{texture index: picture} : the sticker blended on, then the UV layer (borders / checker) if asked for.
        Cached: turning or zooming the car does not blend the sticker again."""
        uvm = self.v_uvcar.get()
        key = (id(self.model), self._sver, self.v_tex.get(), self.v_mode.get(), bool(self.v_flip.get()), uvm,
               round(float(self.v_uvop.get()), 3),
               tuple(c[8] for c in self.visible_chunks()) if uvm == "Borders" else None)
        if key == self._ovr_key:
            return self._ovr
        self._ovr, self._ovr_key = self._textures_override(), key
        return self._ovr

    def _textures_override(self):
        names = [n for n, _ in self.model[0]]
        override = {}
        if self.v_tex.get() not in names:
            return override
        ti = names.index(self.v_tex.get())
        tex = self.model[0][ti][1]
        mode = self.v_mode.get()
        im = {"edited": self.sticker, "original": self.original}.get(mode)
        if im is not None:
            tex = blend_sticker(tex, im, self.v_flip.get())
        uvm = self.v_uvcar.get()
        if uvm != "Off":
            tex = self.uv_on(tex, ti, uvm)
        if tex is not self.model[0][ti][1] or PAINT.match(names[ti]):
            # a paint copy ("p2:<name>") stands in for the base texture "<name>" in the game (wpk_paint): the car's
            # geometry points at the base texture, so the picture of the chosen paint goes in under the base index
            m = PAINT.match(names[ti])
            if m:
                bi = find_texture(names, m.group(2), 0)
                if bi is not None:
                    override[bi] = tex
            else:
                override[ti] = tex
        return override

    def uv_on(self, tex, ti, mode):
        """tex (h x w x 4) with the car's UV panel borders / a UV checker drawn in texture space"""
        h, w = tex.shape[:2]
        op = float(self.v_uvop.get())
        out = tex.astype(np.float32)
        if mode == "Checker":
            c = (((np.arange(h)[:, None] // max(h // 16, 1)) + (np.arange(w)[None, :] // max(w // 16, 1))) % 2)[..., None]
            k = 0.55 * op
            out[..., :3] = np.where(c == 1, out[..., :3] * (1 - k) + 255 * k, out[..., :3] * (1 - k))
        else:
            stem = _stem(PAINT.sub(r"\2", self.model[0][ti][0]))
            chunks = self.visible_chunks()
            key = (self.path, stem, tuple(c[8] for c in chunks), w)
            if key not in self._uvc:
                try:
                    seg = uv_segments(self.model, stem, chunks).copy()
                    seg[:, [1, 3]] = 1 - seg[:, [1, 3]]                    # texture rows run the other way than a PNG
                    lay = draw_segments(seg, w, width=2 if w <= 1024 else 3, color=(255, 0, 90, 255))
                    self._uvc[key] = np.asarray(lay, np.float32)
                except WpkError:
                    self._uvc[key] = None
            lay = self._uvc[key]
            if lay is not None:
                a = lay[..., 3:4] / 255 * op
                out[..., :3] = out[..., :3] * (1 - a) + lay[..., :3] * a
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def show(self):
        if not self.alive() or not self.model:
            return
        gb = self.gbufs.get(self.v_view.get())
        if gb is None:
            if self.photo is None:
                self.canvas.delete("all")
                self.canvas.create_text(self.W // 2, self.H // 2, text="drawing this view …", fill="white")
            else:                                              # keep what is there, just say something is coming
                self.canvas.delete("msg")
                self.canvas.create_text(8, 6, anchor="nw", text="drawing …", fill="white", tags=("msg",))
            self.render_view(self.v_view.get())
            return
        self.paint(gb)

    def compose_now(self, gb):
        return compose(gb, self.model[0], self.textures_override(), self.v_light.get(),
                       REFLECTIONS.get(self.v_refl.get(), 1.0), bool(self.v_night.get()),
                       bool(self.v_brake.get()), bool(self.v_lamps.get()))

    def export_tex(self):
        """every texture inside the car's .wpk (lamps, plate, body paints, wheels ...) as PNG files into a folder"""
        if not self.model:
            return
        from tkinter import filedialog
        d = filedialog.askdirectory(parent=self.top, title="Folder for the car's textures")
        if not d:
            return
        base = os.path.splitext(os.path.basename(self.path or "car"))[0]
        try:
            files = export_textures(self.model, os.path.join(d, base + "_textures"))
            self.msg.set("Wrote %d textures to %s" % (len(files), os.path.join(d, base + "_textures")))
        except Exception as ex:
            self.msg.set("Cannot export the textures: %s" % ex)

    def screenshot(self):
        """save the picture as it is (full size, current view, paint, lighting, reflection, night ...) as a PNG"""
        gb = self.gbufs.get(self.v_view.get()) if self.model else None
        if gb is None:
            self.msg.set("Wait until the view is drawn, then take the screenshot.")
            return
        from tkinter import filedialog
        base = os.path.splitext(os.path.basename(self.path or "car"))[0]
        view = "free" if self.v_view.get() == FREE else VIEWS[self.v_view.get()][0].lower().replace(" ", "").replace("/", "")
        tag = "_night" if self.v_night.get() else ""
        f = filedialog.asksaveasfilename(parent=self.top, title="Save screenshot", defaultextension=".png",
                                         initialfile="%s_%s%s.png" % (base, view, tag),
                                         filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not f:
            return
        try:
            pil = self.compose_now(gb)
            pil.save(f)
            self.msg.set("Saved %s (%d x %d). Maximize the window first for a bigger picture." %
                         (os.path.basename(f), pil.width, pil.height))
        except Exception as ex:
            self.msg.set("Cannot save the screenshot: %s" % ex)

    def remember(self):
        """write the window size and the lighting / reflection / night / lamp choices to the settings file (soon)"""
        if self._pjob is None and self.alive():
            self._pjob = self.top.after(400, self._remember_now)

    def _remember_now(self):
        self._pjob = None
        if not self.alive():
            return
        d = dict(light=self.v_light.get(), reflection=self.v_refl.get(), night=bool(self.v_night.get()),
                 brake=bool(self.v_brake.get()), lamps=bool(self.v_lamps.get()))
        if self.top.state() == "normal":                        # not maximized: that size is the one to come back to
            d.update(width=self.top.winfo_width(), height=self.top.winfo_height())
        save_prefs(preview=d)

    def paint(self, gb):
        pil = self.compose_now(gb)
        if pil.width != self.W:
            pil = pil.resize((self.W, self.H), Image.BILINEAR)         # a quick low-resolution frame while turning
        self.photo = self._ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
