"""Fit a picture to the car's UV layout.

You pin a feature of your art (the roof circle, a door panel ...) and drag the pin to where that feature belongs on the
car's real panel borders. The picture bends between the pins (thin-plate spline), live, and the result is a sticker
that sits 1:1 on the car.

Pure functions (tps, warp) are separate from the window, so they can be used and tested without tkinter.
"""
import numpy as np
from PIL import Image

_E = [i / 6 for i in range(7)]
CORNERS = ([(x, 0.0) for x in _E] + [(x, 1.0) for x in _E] + [(0.0, y) for y in _E[1:-1]] + [(1.0, y) for y in _E[1:-1]])   # the picture's edge


def _kernel(d2):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d2 > 0, 0.5 * d2 * np.log(d2), 0.0)       # r^2 log r


def inverse_map(pins, corners=True):
    """pins: [((src_x, src_y), (dst_x, dst_y)), ...] in 0..1.  Returns f(points n x 2) -> source points: for a pixel of the
    result (dst space) where to read the original picture (src space). f(dst_i) = src_i exactly (pins), and
    the picture's edge points stay where they are when `corners` is on."""
    dst = [np.array(d, float) for _s, d in pins]
    src = [np.array(s, float) for s, _d in pins]
    if corners:
        for c in CORNERS:
            if all(np.hypot(*(np.array(c) - d)) > 0.05 for d in dst):
                dst.append(np.array(c, float))
                src.append(np.array(c, float))
    if not pins:
        return lambda p: np.asarray(p, float)
    dst, src = np.array(dst), np.array(src)
    n = len(dst)
    if n < 3:                                                      # one or two pins and no corners: just move it
        off = (src - dst).mean(0)
        return lambda p: np.asarray(p, float) + off
    K = _kernel(((dst[:, None, :] - dst[None, :, :]) ** 2).sum(-1)) + 1e-4 * np.eye(n)
    P = np.hstack([np.ones((n, 1)), dst])
    L = np.zeros((n + 3, n + 3))
    L[:n, :n], L[:n, n:], L[n:, :n] = K, P, P.T
    Y = np.zeros((n + 3, 2))
    Y[:n] = src
    sol = np.linalg.lstsq(L, Y, rcond=None)[0]
    w, a = sol[:n], sol[n:]

    def f(p):
        p = np.asarray(p, float)
        k = _kernel(((p[:, None, :] - dst[None, :, :]) ** 2).sum(-1))
        return k @ w + a[0] + p @ a[1:]
    return f


def warp(img, pins, size, corners=True, grid=32):
    """img (square or not, stretched to the sticker) bent by the pins -> RGBA size x size"""
    img = img.convert("RGBA")
    if not pins:
        return img.resize((size, size), Image.LANCZOS)
    f = inverse_map(pins, corners)
    g = np.linspace(0, 1, grid + 1)
    gx, gy = np.meshgrid(g, g)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    sp = f(pts).reshape(grid + 1, grid + 1, 2) * np.array(img.size)
    edges = np.round(g * size).astype(int)
    data = []
    for j in range(grid):
        for i in range(grid):
            box = (edges[i], edges[j], edges[i + 1], edges[j + 1])
            nw, ne, sw, se = sp[j, i], sp[j, i + 1], sp[j + 1, i], sp[j + 1, i + 1]
            data.append((box, (*nw, *sw, *se, *ne)))
    return img.transform((size, size), Image.MESH, data, Image.BILINEAR)


class FitWindow:
    """pin-and-drag window. art: PIL image; seg: UV border segments (wpk_view.uv_segments); on_apply(PIL RGBA)"""
    S = 768

    def __init__(self, root, art, seg, title, on_apply, draw_segments):
        import tkinter as tk
        from tkinter import ttk
        from PIL import ImageTk
        self._tk, self._ImageTk = tk, ImageTk
        self.on_apply, self.seg, self.draw_segments = on_apply, seg, draw_segments
        self.art_full = art.convert("RGBA")
        self.art = self.art_full.resize((self.S, self.S), Image.LANCZOS)          # what is bent while you drag
        self.pins = []                                                           # [[src(x, y), dst(x, y)], ...] in 0..1
        self.grab = None
        self.photo = None
        self.v_uv = tk.BooleanVar(value=True)
        self.v_orig = tk.BooleanVar(value=False)
        self.v_corners = tk.BooleanVar(value=True)
        self.v_art = tk.DoubleVar(value=1.0)
        self.msg = tk.StringVar(value="Click the art on a feature, then drag the pin to where it belongs on the red panel borders.")
        self.top = tk.Toplevel(root)
        self.top.title(title)
        self.top.resizable(False, False)
        bar = ttk.Frame(self.top, padding=(8, 6))
        bar.pack(fill="x")
        ttk.Checkbutton(bar, text="Show UV", variable=self.v_uv, command=self.refresh).pack(side="left")
        ttk.Checkbutton(bar, text="Show original art", variable=self.v_orig, command=self.refresh).pack(side="left", padx=8)
        ttk.Checkbutton(bar, text="Keep the edges fixed", variable=self.v_corners, command=self.refresh).pack(side="left")
        ttk.Label(bar, text="Art").pack(side="left", padx=(10, 2))
        sc = ttk.Scale(bar, from_=0.15, to=1.0, variable=self.v_art, length=90, command=lambda _v: self.refresh())
        sc.pack(side="left")
        self.cv = tk.Canvas(self.top, width=self.S, height=self.S, highlightthickness=0, background="#444")
        self.cv.pack()
        self.cv.bind("<ButtonPress-1>", self.press)
        self.cv.bind("<B1-Motion>", self.move)
        self.cv.bind("<ButtonRelease-1>", self.release)
        self.cv.bind("<ButtonPress-3>", self.remove)
        self.cv.bind("<Button-2>", self.remove)
        row = ttk.Frame(self.top, padding=(8, 6))
        row.pack(fill="x")
        ttk.Button(row, text="Undo pin", command=self.undo).pack(side="left")
        ttk.Button(row, text="Clear pins", command=self.clear).pack(side="left", padx=6)
        ttk.Button(row, text="Close", command=self.top.destroy).pack(side="right")
        ttk.Button(row, text="Save PNG…", command=self.save).pack(side="right", padx=6)
        ttk.Button(row, text="Apply to the sticker", command=self.apply).pack(side="right")
        ttk.Label(self.top, textvariable=self.msg, foreground="#555", wraplength=self.S, justify="left",
                  padding=(8, 0, 8, 8)).pack(anchor="w")
        self.uv_img = draw_segments(seg, self.S, width=2)
        self.refresh()

    def alive(self):
        try:
            return bool(self.top.winfo_exists())
        except self._tk.TclError:
            return False

    # ---- pins
    def _pairs(self):
        return [(tuple(s), tuple(d)) for s, d in self.pins]

    def _near(self, x, y, r=12):
        best = None
        for i, (_s, d) in enumerate(self.pins):
            dist = np.hypot(d[0] * self.S - x, d[1] * self.S - y)
            if dist <= r and (best is None or dist < best[0]):
                best = (dist, i)
        return None if best is None else best[1]

    def press(self, e):
        i = self._near(e.x, e.y)
        if i is None:                                   # new pin on whatever is under the cursor now
            f = inverse_map(self._pairs(), self.v_corners.get())
            src = f(np.array([[e.x / self.S, e.y / self.S]]))[0]
            self.pins.append([(float(src[0]), float(src[1])), (e.x / self.S, e.y / self.S)])
            i = len(self.pins) - 1
        self.grab = i
        self.refresh()

    def move(self, e):
        if self.grab is None:
            return
        self.pins[self.grab][1] = (min(max(e.x / self.S, 0.0), 1.0), min(max(e.y / self.S, 0.0), 1.0))
        self.refresh()

    def release(self, _e):
        self.grab = None
        self.msg.set("%d pin%s. Right-click a pin to remove it. Apply to the sticker when it fits." %
                     (len(self.pins), "" if len(self.pins) == 1 else "s"))

    def remove(self, e):
        i = self._near(e.x, e.y)
        if i is not None:
            del self.pins[i]
            self.refresh()

    def undo(self):
        if self.pins:
            self.pins.pop()
            self.refresh()

    def clear(self):
        self.pins = []
        self.refresh()

    # ---- picture
    def current(self, size):
        src = self.art_full if size > self.S else self.art
        return warp(src, self._pairs(), size, self.v_corners.get())

    def refresh(self):
        if not self.alive() or not hasattr(self, "uv_img"):
            return
        im = self.art if self.v_orig.get() else self.current(self.S)
        a = float(self.v_art.get())
        if a < 0.99:
            im = im.copy()
            im.putalpha(im.getchannel("A").point(lambda v: int(v * a)))
        bg = Image.new("RGBA", (self.S, self.S), (68, 68, 68, 255))
        bg.alpha_composite(im)
        if self.v_uv.get():
            bg.alpha_composite(self.uv_img)
        self.photo = self._ImageTk.PhotoImage(bg.convert("RGB"))
        self.cv.delete("all")
        self.cv.create_image(0, 0, anchor="nw", image=self.photo)
        for n, (_s, d) in enumerate(self.pins):
            x, y = d[0] * self.S, d[1] * self.S
            self.cv.create_oval(x - 7, y - 7, x + 7, y + 7, outline="white", width=2, fill="#e0007a")
            self.cv.create_text(x + 14, y - 12, text=str(n + 1), fill="white", font=("TkDefaultFont", 10, "bold"))

    def result(self):
        return self.current(1024)

    def apply(self):
        self.on_apply(self.result())
        self.msg.set("Applied to the sticker. You can keep adjusting and apply again.")

    def save(self):
        from tkinter import filedialog
        f = filedialog.asksaveasfilename(parent=self.top, title="Save the fitted picture", defaultextension=".png",
                                         initialfile="fitted.png", filetypes=[("PNG", "*.png")])
        if f:
            self.result().save(f)
            self.msg.set("Saved %s" % f)
