"""
GIS.PY  —  Forest Carbon & Edge-Effect Decision Support System
================================================================
A lightweight desktop tool for simulating deforestation, edge effects,
carbon-stock valuation and habitat connectivity for endemic species
conservation (e.g. Papua).

Dependencies: tkinter (stdlib) + numpy + matplotlib.
    * tkinter    -> UI, interactive habitat canvas, hand-drawn icons
    * numpy      -> spatial grid + vectorised edge-effect analysis
    * matplotlib -> dashboard charts (trend, donut, connectivity gauge)
    * csv/random -> grid import/export and random sampling

Install & run:
    python3 -m pip install numpy matplotlib
    python3 gis.py

Logos (optional, for the intro):
    Place up to two PNG files inside  ./assets/logo/
    (any two .png files are picked up automatically — first
    alphabetically becomes the primary mark, second the partner mark).
    If none are found, a typographic wordmark is used instead.

Scientific basis
-----------------
* Edge biomass degradation coefficient 25-36% (Pfeifer et al., 2017).
* Carbon valuation via IDXCarbon exchange rate (tCO2e -> IDR), REDD+ RBP.

Structure of this file
-----------------------
SECTION 0 — configuration, palette, shared helpers, icon library
SECTION 1 — splash / intro screen (static logos + loading bar)
SECTION 2 — workspace (initialise, paint canvas, simulate controls)
SECTION 3 — ecological dashboard (results, charts, alerts)
"""

import os
import sys
import csv
import math
import time
import random
import tkinter as tk
from tkinter import filedialog, font as tkfont

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================================================
# SECTION 0 — CONFIGURATION / SCIENTIFIC CONSTANTS
# ==========================================================================
GRID_SIZE   = 20                # 20 x 20 = 400 cells
HA_PER_CELL = 1.0               # each cell = 1 hectare

# Cell states
INFRA = 0   # Infrastructure / Deforestation
CORE  = 1   # Inner / Core forest
EDGE  = 2   # Edge forest (exposed boundary)

# Carbon model (tropical rainforest, above + below ground biomass)
CARBON_DENSITY_CORE = 200.0     # tonnes of Carbon per hectare of intact forest
EDGE_DEGRADATION    = 0.31      # 31% loss on exposed edges (Pfeifer et al. 25-36%)
C_TO_CO2            = 44.0 / 12.0   # elemental Carbon -> CO2 equivalent
IDR_PER_TCO2E       = 60000     # IDXCarbon indicative price (IDR per tonne CO2e)

# Conservation thresholds
MIN_VIABLE_CORE_HA  = 50        # smallest viable contiguous core habitat block
SAFE_CORE_FRACTION  = 0.40      # core should stay above 40% of the landscape

# When frozen by PyInstaller, bundled data lives under sys._MEIPASS.
_BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
LOGO_DIR = os.path.join(_BASE_DIR, "assets", "logo")

# --------------------------------------------------------------------------
# DESIGN TOKENS — a restrained, professional dark palette
# --------------------------------------------------------------------------
BG_APP     = "#12151c"
BG_SIDE    = "#171b24"
BG_PANEL   = "#14171f"
BG_CARD    = "#1b1f2a"
BG_CARD_HI = "#232837"
BG_BAR     = "#171a22"
BORDER     = "#2a2f3c"
BORDER_HI  = "#3a4152"

FG_TEXT    = "#e8eaee"
FG_MUTED   = "#9aa1b0"
FG_FAINT   = "#5b6270"

ACCENT     = "#4f7cff"
ACCENT_DK  = "#3b62d6"
ACCENT_SOFT= "#1c2c52"
GREEN      = "#3ea56a"
GREEN_DK   = "#2c7a4d"
AMBER      = "#c98a2c"
AMBER_DK   = "#8f5f1a"
RED        = "#c1524a"
RED_DK     = "#8f3630"
RED_SOFT   = "#3a2222"
SLATE_DK   = "#333846"

COLOR_CORE   = "#3d8a5c"
COLOR_EDGE   = "#b8823a"
COLOR_INFRA  = "#23262e"
GRID_BG      = "#0e1016"

CELL   = 23
GAP    = 2
ORIGIN = 14
GRID_PX = GRID_SIZE * CELL

UI_FONT = "Helvetica"


# --------------------------------------------------------------------------
# Formatting helpers (Indonesian thousands separator = ".")
# --------------------------------------------------------------------------
def fmt_int(v):
    return "{:,.0f}".format(v).replace(",", ".")


def fmt_idr(v):
    return "Rp " + fmt_int(v)


def _pick_font(root):
    global UI_FONT
    try:
        fams = set(tkfont.families(root))
    except Exception:
        return
    for f in ("Segoe UI", "Helvetica Neue", "Helvetica", "Arial"):
        if f in fams:
            UI_FONT = f
            return


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _blend(hex_a, hex_b, t):
    """Linear-interpolate between two #rrggbb colours (t in [0,1])."""
    t = _clamp(t, 0.0, 1.0)
    a = tuple(int(hex_a[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(hex_b[i:i + 2], 16) for i in (1, 3, 5))
    mix = tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return "#%02x%02x%02x" % mix


# --------------------------------------------------------------------------
# Vector icon library — every icon in this app is hand-drawn on a Canvas
# (no image files needed for the UI chrome).
# --------------------------------------------------------------------------
def draw_icon(cv, name, cx, cy, s, color, weight=2):
    """Draw a small vector glyph centred at (cx, cy) with 'radius' s."""
    w = weight
    if name == "forest":
        cv.create_polygon(cx, cy - s, cx - s * 0.75, cy + s * 0.15,
                           cx + s * 0.75, cy + s * 0.15, fill=color, outline="")
        cv.create_polygon(cx, cy - s * 0.45, cx - s * 0.6, cy + s * 0.55,
                           cx + s * 0.6, cy + s * 0.55, fill=color, outline="")
        cv.create_rectangle(cx - s * 0.09, cy + s * 0.5, cx + s * 0.09, cy + s * 0.95,
                             fill=color, outline="")
    elif name == "deforest":
        draw_icon(cv, "forest", cx, cy, s, color)
        cv.create_line(cx - s, cy - s, cx + s, cy + s, fill=RED, width=w + 1)
        cv.create_line(cx - s, cy + s, cx + s, cy - s, fill=RED, width=w + 1)
    elif name == "clear":
        cv.create_oval(cx - s, cy - s, cx + s, cy + s, outline=color, width=w)
        for ang in (0, 60, 120, 180, 240, 300):
            rad = math.radians(ang)
            x1, y1 = cx + s * 0.55 * math.cos(rad), cy + s * 0.55 * math.sin(rad)
            x2, y2 = cx + s * 1.25 * math.cos(rad), cy + s * 1.25 * math.sin(rad)
            cv.create_line(x1, y1, x2, y2, fill=color, width=w)
    elif name == "new":
        cv.create_rectangle(cx - s * 0.6, cy - s, cx + s * 0.6, cy + s,
                             outline=color, width=w)
        cv.create_line(cx, cy - s * 0.35, cx, cy + s * 0.35, fill=color, width=w)
        cv.create_line(cx - s * 0.35, cy, cx + s * 0.35, cy, fill=color, width=w)
    elif name == "import":
        cv.create_line(cx, cy - s, cx, cy + s * 0.25, fill=color, width=w)
        cv.create_polygon(cx - s * 0.45, cy - s * 0.1, cx + s * 0.45, cy - s * 0.1,
                           cx, cy + s * 0.45, fill=color, outline="")
        cv.create_line(cx - s, cy + s * 0.85, cx + s, cy + s * 0.85, fill=color, width=w)
    elif name == "export":
        cv.create_line(cx, cy + s * 0.5, cx, cy - s * 0.35, fill=color, width=w)
        cv.create_polygon(cx - s * 0.45, cy - s * 0.05, cx + s * 0.45, cy - s * 0.05,
                           cx, cy - s * 0.6, fill=color, outline="")
        cv.create_line(cx - s, cy + s * 0.85, cx + s, cy + s * 0.85, fill=color, width=w)
    elif name == "reset":
        cv.create_arc(cx - s, cy - s, cx + s, cy + s, start=40, extent=270,
                       style="arc", outline=color, width=w)
        cv.create_polygon(cx + s * 0.75, cy - s * 0.85, cx + s * 1.15, cy - s * 0.3,
                           cx + s * 0.35, cy - s * 0.35, fill=color, outline="")
    elif name == "play":
        cv.create_polygon(cx - s * 0.6, cy - s, cx - s * 0.6, cy + s, cx + s, cy,
                           fill=color, outline="")
    elif name == "dashboard":
        bars = (-0.6, -0.15, 0.3)
        heights = (0.5, 1.0, 0.7)
        for bx, bh in zip(bars, heights):
            cv.create_rectangle(cx + bx * s, cy + s - 2 * s * bh,
                                 cx + bx * s + s * 0.4, cy + s,
                                 fill=color, outline="")
    elif name == "leaf":
        cv.create_arc(cx - s, cy - s, cx + s, cy + s, start=30, extent=180,
                       style="chord", fill=color, outline="")
        cv.create_line(cx - s * 0.7, cy + s * 0.55, cx + s * 0.75, cy - s * 0.55,
                        fill=BG_CARD, width=w)
    elif name == "warning":
        cv.create_polygon(cx, cy - s, cx + s, cy + s, cx - s, cy + s,
                           outline=color, width=w, fill="")
        cv.create_line(cx, cy - s * 0.25, cx, cy + s * 0.3, fill=color, width=w)
        cv.create_oval(cx - w * 0.6, cy + s * 0.55, cx + w * 0.6, cy + s * 0.55 + w * 1.2,
                        fill=color, outline="")
    elif name == "critical":
        cv.create_oval(cx - s, cy - s, cx + s, cy + s, outline=color, width=w)
        cv.create_line(cx - s * 0.5, cy - s * 0.5, cx + s * 0.5, cy + s * 0.5,
                        fill=color, width=w)
        cv.create_line(cx - s * 0.5, cy + s * 0.5, cx + s * 0.5, cy - s * 0.5,
                        fill=color, width=w)
    elif name == "check":
        cv.create_oval(cx - s, cy - s, cx + s, cy + s, outline=color, width=w)
        cv.create_line(cx - s * 0.45, cy + s * 0.05, cx - s * 0.1, cy + s * 0.45,
                        fill=color, width=w + 1)
        cv.create_line(cx - s * 0.1, cy + s * 0.45, cx + s * 0.55, cy - s * 0.4,
                        fill=color, width=w + 1)
    elif name == "building":
        cv.create_rectangle(cx - s * 0.7, cy - s, cx + s * 0.7, cy + s,
                             outline=color, width=w)
        for ix in (-0.35, 0.35):
            for iy in (-0.55, -0.05, 0.45):
                cv.create_rectangle(cx + ix * s - s * 0.12, cy + iy * s - s * 0.12,
                                     cx + ix * s + s * 0.12, cy + iy * s + s * 0.12,
                                     fill=color, outline="")
    elif name == "cloud":
        cv.create_oval(cx - s, cy - s * 0.2, cx - s * 0.2, cy + s * 0.6, fill=color, outline="")
        cv.create_oval(cx - s * 0.35, cy - s * 0.6, cx + s * 0.55, cy + s * 0.5, fill=color, outline="")
        cv.create_oval(cx + s * 0.1, cy - s * 0.1, cx + s * 0.9, cy + s * 0.6, fill=color, outline="")
        cv.create_rectangle(cx - s, cy + s * 0.15, cx + s * 0.9, cy + s * 0.6, fill=color, outline="")
    elif name == "link":
        cv.create_oval(cx - s, cy - s * 0.4, cx, cy + s * 0.4, outline=color, width=w)
        cv.create_oval(cx, cy - s * 0.4, cx + s, cy + s * 0.4, outline=color, width=w)


# ==========================================================================
# Re-usable rounded-rect / button helpers (shared across all sections)
# ==========================================================================
def round_rect(cv, x1, y1, x2, y2, r, **kw):
    r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return cv.create_polygon(pts, smooth=True, **kw)


def round_button(parent, text, command, fill, hover, fg="white", size=10,
                  bold=True, height=42, radius=9, outline="", bg=BG_SIDE,
                  icon=None):
    cv = tk.Canvas(parent, height=height, bg=bg, highlightthickness=0, cursor="hand2")
    state = {"fill": fill}
    weight = "bold" if bold else "normal"

    def draw(w=None):
        w = int(w) if w else cv.winfo_width()
        if w <= 1:
            return
        cv.delete("all")
        round_rect(cv, 1, 1, w - 1, height - 1, radius, fill=state["fill"],
                   outline=outline, width=1)
        tx = w / 2
        if icon:
            draw_icon(cv, icon, 22, height / 2, 7, fg, weight=2)
            tx = w / 2 + 10
        cv.create_text(tx, height / 2 + 1, text=text, fill=fg,
                        font=(UI_FONT, size, weight))

    cv.bind("<Configure>", lambda e: draw(e.width))
    cv.bind("<Enter>", lambda e: (state.update(fill=hover), draw()))
    cv.bind("<Leave>", lambda e: (state.update(fill=fill), draw()))
    cv.bind("<Button-1>", lambda e: command())
    return cv


def icon_dock_button(parent, name, tooltip, command, size=44):
    """A tool-dock style square icon button with a hover tooltip."""
    cv = tk.Canvas(parent, width=size, height=size, bg=BG_SIDE,
                   highlightthickness=0, cursor="hand2")
    state = {"active": False, "hover": False}
    tip = {"win": None}

    def render():
        cv.delete("all")
        if state["active"]:
            bg, fg = BG_CARD_HI, ACCENT
            round_rect(cv, 3, 3, size - 3, size - 3, 8, fill=bg, outline=ACCENT, width=1)
        elif state["hover"]:
            bg, fg = BG_CARD, FG_TEXT
            round_rect(cv, 3, 3, size - 3, size - 3, 8, fill=bg, outline="")
        else:
            bg, fg = BG_SIDE, FG_MUTED
        draw_icon(cv, name, size / 2, size / 2, size * 0.20, fg, weight=2)

    def show_tip():
        if tip["win"] is not None or not tooltip or not state["hover"]:
            return
        try:
            x = cv.winfo_rootx() + size + 10
            y = cv.winfo_rooty() + size // 2 - 12
        except tk.TclError:
            return
        t = tk.Toplevel(cv)
        t.overrideredirect(True)
        try:
            t.attributes("-topmost", True)
        except tk.TclError:
            pass
        frame = tk.Frame(t, bg=BORDER_HI)
        frame.pack()
        tk.Label(frame, text=tooltip, bg=BG_CARD_HI, fg=FG_TEXT,
                 font=(UI_FONT, 9), padx=9, pady=5).pack(padx=1, pady=1)
        t.geometry("+%d+%d" % (x, y))
        tip["win"] = t

    def hide_tip():
        if tip["win"] is not None:
            try:
                tip["win"].destroy()
            except Exception:
                pass
            tip["win"] = None

    def on_enter(_e):
        state["hover"] = True
        render()
        cv.after(400, show_tip)

    def on_leave(_e):
        state["hover"] = False
        render()
        hide_tip()

    cv.bind("<Enter>", on_enter)
    cv.bind("<Leave>", on_leave)
    cv.bind("<Button-1>", lambda e: (hide_tip(), command()))
    render()
    cv.set_active = lambda v: (state.update(active=v), render())
    return cv


# ==========================================================================
# ALERT SYSTEM — an in-app overlay (see the Alert docstring for why this is
# not a separate modal window).
# ==========================================================================
class Alert:
    """SweetAlert-style alert rendered as an in-app overlay (a Frame placed
    over the main window) instead of a separate modal window.

    Borderless Toplevels with grab_set() are unreliable on BOTH Windows and
    macOS: they can grab all input yet fail to deliver the button click, which
    wedges the whole app (you can't click anything or close it). An overlay is
    an ordinary widget, so its button always fires — no grab_set, no borderless
    window, no per-platform focus quirks. Only one alert shows at a time;
    critical alerts pulse red."""

    ICONS = {"critical": ("critical", RED, RED_SOFT),
             "warning": ("warning", AMBER, "#3a2f18"),
             "healthy": ("check", GREEN, "#173323")}

    _cur = {"overlay": None, "job": None, "root": None}

    @staticmethod
    def _close():
        cur = Alert._cur
        if cur.get("job") is not None and cur.get("root") is not None:
            try:
                cur["root"].after_cancel(cur["job"])
            except Exception:
                pass
        cur["job"] = None
        if cur.get("root") is not None:
            try:
                cur["root"].unbind("<Escape>")
            except Exception:
                pass
        if cur.get("overlay") is not None:
            try:
                cur["overlay"].destroy()
            except Exception:
                pass
        cur["overlay"] = None

    @staticmethod
    def show(parent, level, title, message, confirm_text="Understood"):
        Alert._close()
        icon, color, panel_bg = Alert.ICONS.get(level, Alert.ICONS["warning"])
        root = parent.winfo_toplevel()
        Alert._cur["root"] = root

        # Full-window dim overlay — a real widget, so clicks are delivered
        # normally and no grab is needed. It sits above everything and
        # swallows backdrop clicks, giving modal behaviour without the pitfalls.
        overlay = tk.Frame(root, bg="#0a0d14")
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        overlay.lift()
        overlay.bind("<Button-1>", lambda e: "break")
        Alert._cur["overlay"] = overlay

        card = tk.Frame(overlay, bg=BG_CARD, highlightbackground=color,
                        highlightthickness=2)
        card.place(relx=0.5, rely=0.5, anchor="center")

        ring = tk.Canvas(card, width=104, height=104, bg=BG_CARD,
                         highlightthickness=0)
        ring.pack(pady=(28, 4), padx=44)
        tk.Label(card, text=title, font=(UI_FONT, 15, "bold"), fg=FG_TEXT,
                 bg=BG_CARD).pack(pady=(2, 6))
        tk.Label(card, text=message, font=(UI_FONT, 10), fg=FG_MUTED, bg=BG_CARD,
                 wraplength=320, justify="center").pack(padx=28)

        def draw_ring(col):
            ring.delete("all")
            ring.create_oval(15, 15, 89, 89, fill=panel_bg, outline=col, width=3)
            draw_icon(ring, icon, 52, 52, 19, col, weight=3)

        draw_ring(color)

        wrap = tk.Frame(card, bg=BG_CARD)
        wrap.pack(fill="x", padx=28, pady=24)
        round_button(wrap, confirm_text, Alert._close, color,
                     _blend(color, "#ffffff", 0.14), fg="#ffffff", size=11,
                     height=42, radius=8, bg=BG_CARD).pack(fill="x")

        root.bind("<Escape>", lambda e: Alert._close())

        if level == "critical":
            state = {"phase": 0.0}

            def pulse():
                if Alert._cur.get("overlay") is not overlay:
                    return
                state["phase"] += 0.24
                k = 0.5 + 0.5 * math.sin(state["phase"])
                col = _blend(RED_DK, "#ff5347", k)
                draw_ring(col)
                try:
                    card.config(highlightbackground=col)
                except Exception:
                    pass
                Alert._cur["job"] = root.after(45, pulse)

            pulse()


# ==========================================================================
# MAIN APPLICATION
# ==========================================================================
class GisPyApp:
    def __init__(self, root):
        self.root = root
        _pick_font(root)
        self.root.title("GIS.PY — Forest Carbon & Edge-Effect DSS")
        self.root.geometry("1360x880")
        self.root.configure(bg=BG_APP)
        self.root.minsize(1280, 780)

        # Spatial matrix — a numpy array (rows × cols) of cell states
        self.forest_map = np.full((GRID_SIZE, GRID_SIZE), CORE, dtype=int)
        self.rects = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]

        self.history_core_pct = [100.0]

        self.paint_mode = tk.IntVar(value=CORE)
        self.brush_size = tk.IntVar(value=1)
        self.clear_pct = tk.IntVar(value=15)
        self._hover = None
        self.metric_vars = {}
        self._tool_btns = {}
        self._last_stats = None

        # Root container: everything (menu, workspace, dashboard) lives here.
        # The splash screen is drawn *over* it and destroyed once ready.
        self.chrome = tk.Frame(self.root, bg=BG_APP)
        self.content = tk.Frame(self.root, bg=BG_APP)

        self._build_menubar()
        self._build_workspace()
        self._build_dashboard()
        self.show_workspace()

        self.start_blank()

        # Everything above is built off-screen while the splash plays.
        self.chrome.pack_forget()
        self.content.pack_forget()
        Splash(self.root, on_finished=self._reveal_app)

    def _reveal_app(self):
        self.chrome.pack(side="top", fill="x")
        self.content.pack(fill="both", expand=True)
        # Force a full geometry pass now that everything is mapped, so any
        # size-dependent drawing (canvases, charts) reflects real dimensions
        # immediately instead of only after the window is later resized.
        self.root.update_idletasks()

    def _card(self, parent, bg=BG_CARD, border=BORDER):
        return tk.Frame(parent, bg=bg, highlightbackground=border,
                        highlightthickness=1, bd=0)

    def _mpl_fig(self, parent, bg):
        """A dark-themed matplotlib figure embedded in a Tk parent.
        Returns (figure, axes, tk_widget)."""
        fig = Figure(figsize=(3.2, 2.2), dpi=100, facecolor=bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        widget = canvas.get_tk_widget()
        widget.configure(bg=bg, highlightthickness=0)
        return fig, ax, widget

    def _load_brand_logo(self):
        """Load the app (snake) mark for the top-left brand, downscaled small."""
        try:
            if not os.path.isdir(LOGO_DIR):
                return None
            files = sorted(f for f in os.listdir(LOGO_DIR)
                           if f.lower().endswith(".png"))
            if not files:
                return None
            pick = next((f for f in files
                         if "gis" in f.lower() or "snake" in f.lower()), None)
            if pick is None:
                pick = files[1] if len(files) >= 2 else files[0]
            img = tk.PhotoImage(file=os.path.join(LOGO_DIR, pick))
            big = max(img.width(), img.height())
            if big > 26:
                img = img.subsample(max(1, round(big / 24)))
            return img
        except Exception:
            return None

    # ======================================================================
    # SECTION 2A — MENU BAR
    # ======================================================================
    def _build_menubar(self):
        bar = tk.Frame(self.chrome, bg=BG_BAR, height=40)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        brand = tk.Frame(bar, bg=BG_BAR)
        brand.pack(side="left", padx=(16, 4))
        self._brand_img = self._load_brand_logo()
        if self._brand_img is not None:
            tk.Label(brand, image=self._brand_img, bg=BG_BAR).pack(side="left", pady=6)
        else:
            logo = tk.Canvas(brand, width=18, height=18, bg=BG_BAR, highlightthickness=0)
            logo.pack(side="left", pady=11)
            logo.create_rectangle(1, 1, 8, 8, fill=COLOR_CORE, outline="")
            logo.create_rectangle(10, 1, 17, 8, fill=COLOR_EDGE, outline="")
            logo.create_rectangle(1, 10, 8, 17, fill=COLOR_EDGE, outline="")
            logo.create_rectangle(10, 10, 17, 17, fill=COLOR_INFRA, outline="")
        tk.Label(brand, text="GIS.PY", fg=FG_TEXT, bg=BG_BAR,
                 font=(UI_FONT, 10, "bold")).pack(side="left", padx=(8, 14))

        def menubutton(label, items):
            # Custom high-contrast menu button. (tk.Menubutton renders as a
            # washed-out native control on macOS and ignores our colours, so
            # we use a Label and pop the menu manually.)
            menu = tk.Menu(bar, tearoff=0, bg=BG_CARD, fg=FG_TEXT,
                          activebackground=ACCENT, activeforeground="#ffffff",
                          bd=0, relief="flat", font=(UI_FONT, 10))
            for it in items:
                if it == "---":
                    menu.add_separator()
                else:
                    txt, cmd = it
                    menu.add_command(label=txt, command=cmd)
            lbl = tk.Label(bar, text=label, bg=BG_BAR, fg=FG_TEXT,
                           font=(UI_FONT, 10, "bold"), padx=12, cursor="hand2")

            def popup(_e):
                try:
                    menu.tk_popup(lbl.winfo_rootx(),
                                  lbl.winfo_rooty() + lbl.winfo_height())
                finally:
                    menu.grab_release()

            lbl.bind("<Button-1>", popup)
            lbl.bind("<Enter>", lambda e: lbl.config(bg=BG_CARD_HI))
            lbl.bind("<Leave>", lambda e: lbl.config(bg=BG_BAR))
            lbl.pack(side="left", fill="y")
            return lbl

        menubutton("File", [
            ("New Blank Canvas", lambda: self.start_blank()),
            ("Import Grid (.csv)…", self.import_csv),
            ("Export Grid (.csv)…", self.export_csv),
            "---",
            ("Exit", self.root.destroy),
        ])
        menubutton("Edit", [
            ("Reset Forest", lambda: self.start_blank()),
            ("Apply Random Clearing", self.apply_random_deforestation),
        ])
        menubutton("View", [
            ("Workspace", self.show_workspace),
            ("Dashboard", self.show_dashboard),
        ])
        menubutton("Simulation", [
            ("Run Simulation", self.simulate),
        ])
        menubutton("Help", [
            ("About / Scientific Basis", self._show_about),
        ])

        tk.Label(bar, text="Decision Support System · v1.0", fg=FG_FAINT, bg=BG_BAR,
                font=(UI_FONT, 8)).pack(side="right", padx=16)

    def _show_about(self):
        Alert.show(self.root, "healthy", "Scientific Basis",
                   "Edge biomass degradation: 25-36% (Pfeifer et al., 2017), "
                   "modelled here at 31%. Carbon valuation follows the "
                   "IDXCarbon indicative rate under REDD+ Result-Based "
                   "Payments. 1 grid cell = 1 hectare.",
                   confirm_text="Close")

    # ======================================================================
    # SECTION 2 — WORKSPACE  (initialise · paint canvas · run controls)
    # ======================================================================
    def _build_workspace(self):
        self.workspace_frame = tk.Frame(self.content, bg=BG_APP)
        self.workspace_frame.place(x=0, y=0, relwidth=1, relheight=1)

        # ---- left icon dock ----
        dock = tk.Frame(self.workspace_frame, bg=BG_SIDE, width=64)
        dock.pack(side="left", fill="y")
        dock.pack_propagate(False)

        tk.Label(dock, text="TOOLS", font=(UI_FONT, 7, "bold"), fg=FG_FAINT,
                 bg=BG_SIDE).pack(pady=(14, 6))

        b_forest = icon_dock_button(dock, "forest", "Plant inner forest",
                                    lambda: self._set_tool(CORE))
        b_forest.pack(pady=4)
        b_deforest = icon_dock_button(dock, "deforest", "Clear / deforest",
                                      lambda: self._set_tool(INFRA))
        b_deforest.pack(pady=4)
        self._tool_btns = {CORE: b_forest, INFRA: b_deforest}

        tk.Frame(dock, bg=BORDER, height=1, width=40).pack(pady=12)

        icon_dock_button(dock, "new", "New blank canvas",
                         lambda: self.start_blank()).pack(pady=4)
        icon_dock_button(dock, "import", "Import grid (.csv)",
                         self.import_csv).pack(pady=4)
        icon_dock_button(dock, "export", "Export grid (.csv)",
                         self.export_csv).pack(pady=4)

        # ---- center column: options bar + canvas + bottom action bar ----
        center = tk.Frame(self.workspace_frame, bg=BG_APP)
        center.pack(side="left", fill="both", expand=True)

        opt = tk.Frame(center, bg=BG_BAR, height=42)
        opt.pack(fill="x")
        opt.pack_propagate(False)
        self.opt_tool_label = tk.Label(opt, text="FOREST BRUSH", font=(UI_FONT, 9, "bold"),
                                       fg=ACCENT, bg=BG_BAR)
        self.opt_tool_label.pack(side="left", padx=(16, 18))
        tk.Label(opt, text="Brush size", font=(UI_FONT, 9), fg=FG_MUTED,
                 bg=BG_BAR).pack(side="left")
        seg = tk.Frame(opt, bg=BG_BAR)
        seg.pack(side="left", padx=8)
        self._brush_seg_btns = {}
        for n in (1, 2, 3):
            cv = tk.Canvas(seg, width=26, height=26, bg=BG_BAR, highlightthickness=0,
                          cursor="hand2")
            cv.create_text(13, 13, text=str(n), fill=FG_MUTED, font=(UI_FONT, 9, "bold"))
            cv.bind("<Button-1>", lambda e, n=n: self._set_brush(n))
            cv.pack(side="left", padx=2)
            self._brush_seg_btns[n] = cv
        self._set_brush(1)
        tk.Label(opt, text="1 grid = 1 hectare  ·  click & drag to paint",
                 font=(UI_FONT, 8), fg=FG_FAINT, bg=BG_BAR).pack(side="right", padx=16)

        # canvas area
        canvas_wrap = tk.Frame(center, bg=BG_APP)
        canvas_wrap.pack(fill="both", expand=True)

        head = tk.Frame(canvas_wrap, bg=BG_APP)
        head.pack(anchor="w", padx=22, pady=(16, 8), fill="x")
        tk.Label(head, text="Interactive Habitat Canvas", font=(UI_FONT, 13, "bold"),
                 fg=FG_TEXT, bg=BG_APP).pack(anchor="w")
        row = tk.Frame(head, bg=BG_APP)
        row.pack(anchor="w", fill="x")
        tk.Label(row, text="20 × 20 grid  ·  400 hectares", font=(UI_FONT, 9),
                 fg=FG_MUTED, bg=BG_APP).pack(side="left")
        self.live_readout = tk.StringVar(value="")
        tk.Label(row, textvariable=self.live_readout, font=(UI_FONT, 9),
                 fg=FG_FAINT, bg=BG_APP).pack(side="right")

        size = GRID_PX + 2 * ORIGIN
        shell = tk.Frame(canvas_wrap, bg=BG_CARD, highlightbackground=BORDER,
                         highlightthickness=1)
        shell.pack()
        self.map_canvas = tk.Canvas(shell, width=size, height=size, bg=GRID_BG,
                                    highlightthickness=0, cursor="crosshair")
        self.map_canvas.pack(padx=10, pady=10)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x0 = ORIGIN + c * CELL + GAP
                y0 = ORIGIN + r * CELL + GAP
                self.rects[r][c] = self.map_canvas.create_rectangle(
                    x0, y0, x0 + CELL - GAP, y0 + CELL - GAP,
                    fill=COLOR_CORE, outline="", width=2)
        self.map_canvas.bind("<Button-1>", self.on_paint)
        self.map_canvas.bind("<B1-Motion>", self.on_paint)
        self.map_canvas.bind("<Motion>", self._on_hover)
        self.map_canvas.bind("<Leave>", self._leave_grid)

        legend = tk.Frame(canvas_wrap, bg=BG_APP)
        legend.pack(pady=(12, 0))
        for colour, label in ((COLOR_CORE, "Inner Forest"),
                              (COLOR_EDGE, "Edge Forest"),
                              (COLOR_INFRA, "Deforested")):
            chip = tk.Frame(legend, bg=BG_CARD, highlightbackground=BORDER,
                            highlightthickness=1)
            chip.pack(side="left", padx=5)
            tk.Frame(chip, bg=colour, width=11, height=11).pack(side="left", padx=(8, 6), pady=6)
            tk.Label(chip, text=label, fg=FG_MUTED, bg=BG_CARD,
                     font=(UI_FONT, 8)).pack(side="left", padx=(0, 10))

        # ---- bottom action bar ----
        bottom = tk.Frame(center, bg=BG_BAR, height=74)
        bottom.pack(side="bottom", fill="x")
        bottom.pack_propagate(False)
        inner = tk.Frame(bottom, bg=BG_BAR)
        inner.pack(fill="both", expand=True, padx=18, pady=13)

        round_button(inner, "Reset", lambda: self.start_blank(), BG_CARD,
                    BG_CARD_HI, fg=FG_MUTED, bold=False, size=9, height=44,
                    icon="reset", bg=BG_BAR).pack(side="left", padx=(0, 8))

        # compact single-row clearing stepper (fits the bar height)
        clear_wrap = tk.Frame(inner, bg=BG_CARD, highlightbackground=BORDER,
                              highlightthickness=1)
        clear_wrap.pack(side="left", padx=(0, 8))
        cin = tk.Frame(clear_wrap, bg=BG_CARD)
        cin.pack(padx=10, pady=9)
        tk.Label(cin, text="Clearing", font=(UI_FONT, 8), fg=FG_FAINT,
                 bg=BG_CARD).pack(side="left", padx=(0, 8))

        def _stepper(txt, delta):
            b = tk.Label(cin, text=txt, font=(UI_FONT, 12, "bold"), fg=FG_TEXT,
                         bg=BG_CARD_HI, width=2, cursor="hand2")
            b.bind("<Button-1>", lambda e: self._nudge_clear(delta))
            b.bind("<Enter>", lambda e: b.config(bg=BORDER_HI))
            b.bind("<Leave>", lambda e: b.config(bg=BG_CARD_HI))
            b.pack(side="left", padx=1)

        _stepper("–", -5)
        self.clear_val_lbl = tk.Label(cin, text="15%", font=(UI_FONT, 10, "bold"),
                                      fg=ACCENT, bg=BG_CARD, width=4)
        self.clear_val_lbl.pack(side="left")
        _stepper("+", 5)

        round_button(inner, "Random", self.apply_random_deforestation,
                    RED_DK, RED, size=9, height=44, icon="clear",
                    bg=BG_BAR).pack(side="left", padx=(0, 8))

        round_button(inner, "SIMULATE", self.simulate, GREEN_DK, GREEN, size=11,
                    height=44, icon="play", bg=BG_BAR).pack(side="left", padx=(0, 8))

        round_button(inner, "Dashboard", self.show_dashboard, ACCENT_DK, ACCENT,
                    fg="#ffffff", size=10, height=44, icon="dashboard",
                    bg=BG_BAR).pack(side="right")

    def _nudge_clear(self, delta):
        v = _clamp(self.clear_pct.get() + delta, 5, 80)
        self.clear_pct.set(v)
        self.clear_val_lbl.config(text="%d%%" % v)

    def _set_tool(self, val):
        self.paint_mode.set(val)
        for v, btn in self._tool_btns.items():
            btn.set_active(v == val)
        self.opt_tool_label.config(
            text="FOREST BRUSH" if val == CORE else "DEFOREST BRUSH",
            fg=GREEN if val == CORE else RED)

    def _set_brush(self, n):
        self.brush_size.set(n)
        for k, cv in self._brush_seg_btns.items():
            on = (k == n)
            cv.delete("all")
            if on:
                round_rect(cv, 1, 1, 25, 25, 6, fill=ACCENT_SOFT, outline=ACCENT)
            cv.create_text(13, 13, text=str(k), fill=(ACCENT if on else FG_MUTED),
                           font=(UI_FONT, 9, "bold"))

    # -------------------------------------------------------------- PAINTING
    def on_paint(self, event):
        c = (event.x - ORIGIN) // CELL
        r = (event.y - ORIGIN) // CELL
        b = self.brush_size.get()
        state = self.paint_mode.get()
        changed = False
        for dr in range(b):
            for dc in range(b):
                rr, cc = r + dr, c + dc
                if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE:
                    if self.forest_map[rr][cc] != state:
                        self.forest_map[rr][cc] = state
                        changed = True
        if changed:
            self.compute_edges()
            self.render_grid()
            self._update_live_readout()

    def _on_hover(self, event):
        c = (event.x - ORIGIN) // CELL
        r = (event.y - ORIGIN) // CELL
        cell = (r, c) if (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE) else None
        if cell == self._hover:
            return
        if self._hover:
            pr, pc = self._hover
            self.map_canvas.itemconfig(self.rects[pr][pc], outline="")
        if cell:
            self.map_canvas.itemconfig(self.rects[r][c], outline=ACCENT)
        self._hover = cell

    def _leave_grid(self, event):
        if self._hover:
            pr, pc = self._hover
            self.map_canvas.itemconfig(self.rects[pr][pc], outline="")
            self._hover = None

    def _update_live_readout(self):
        core = int(np.count_nonzero(self.forest_map == CORE))
        edge = int(np.count_nonzero(self.forest_map == EDGE))
        infra = GRID_SIZE * GRID_SIZE - core - edge
        self.live_readout.set("Core %d ha   ·   Edge %d ha   ·   Cleared %d ha"
                              % (core, edge, infra))

    # ----------------------------------------------------------- SCENARIOS
    def start_blank(self):
        self.forest_map = np.full((GRID_SIZE, GRID_SIZE), CORE, dtype=int)
        self.history_core_pct = [100.0]
        self.render_grid()
        self._update_live_readout()
        self._last_stats = self._analyse()
        if self.dashboard_frame.winfo_ismapped():
            self._update_dashboard(self._last_stats)

    def apply_random_deforestation(self):
        pct = self.clear_pct.get() / 100.0
        standing = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)
                    if self.forest_map[r][c] != INFRA]
        if not standing:
            Alert.show(self.root, "warning", "Nothing to clear",
                      "The forest is already fully cleared.", "OK")
            return
        n = int(len(standing) * pct)
        for (r, c) in random.sample(standing, min(n, len(standing))):
            self.forest_map[r][c] = INFRA
        self.compute_edges()
        self.render_grid()
        self._update_live_readout()

    # ------------------------------------------------------- FILE I/O (CSV)
    def import_csv(self):
        path = filedialog.askopenfilename(
            title="Import a 20x20 grid (.csv)  ·  0=deforested 1=inner 2=edge",
            filetypes=[("CSV grid", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            rows = []
            with open(path, newline="") as f:
                for record in csv.reader(f):
                    cells = [c.strip() for c in record if c.strip() != ""]
                    if cells:
                        rows.append([int(float(c)) for c in cells])
        except (ValueError, OSError) as exc:
            Alert.show(self.root, "critical", "Import failed",
                      "Could not read the CSV. Expected a 20x20 grid of "
                      "integers (0, 1 or 2).\n\n%s" % exc, "OK")
            return
        if len(rows) != GRID_SIZE or any(len(r) != GRID_SIZE for r in rows):
            Alert.show(self.root, "critical", "Wrong grid size",
                      "CSV must be exactly %d x %d cells (got %d rows)."
                      % (GRID_SIZE, GRID_SIZE, len(rows)), "OK")
            return
        if any(v not in (INFRA, CORE, EDGE) for row in rows for v in row):
            Alert.show(self.root, "critical", "Invalid values",
                      "Cells must be 0 (deforested), 1 (inner forest) or 2 (edge).",
                      "OK")
            return

        self.forest_map = np.array(rows, dtype=int)
        self.compute_edges()
        core = int(np.count_nonzero(self.forest_map == CORE))
        self.history_core_pct = [core / (GRID_SIZE * GRID_SIZE) * 100.0]
        self.render_grid()
        self._update_live_readout()
        self._update_dashboard(self._analyse())

    def export_csv(self):
        path = filedialog.asksaveasfilename(
            title="Export current grid as .csv", defaultextension=".csv",
            filetypes=[("CSV grid", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerows(self.forest_map.tolist())
        except OSError as exc:
            Alert.show(self.root, "critical", "Save failed", str(exc), "OK")
            return
        Alert.show(self.root, "healthy", "Grid exported",
                  "Saved to:\n%s" % path, "OK")

    # --------------------------------------------------------- CORE LOGIC
    def compute_edges(self):
        """Edge-Effect Analytics (NumPy): a forest cell exposed to any cleared
        cell in its 8-neighbourhood becomes EDGE. Vectorised with array shifts
        instead of Python loops."""
        m = self.forest_map
        m[m != INFRA] = CORE                       # reset standing forest to core
        infra = (m == INFRA)
        exposed = np.zeros_like(m, dtype=bool)
        # OR the cleared mask shifted in from each of the 8 directions
        exposed[1:, :]   |= infra[:-1, :]          # neighbour above
        exposed[:-1, :]  |= infra[1:, :]           # below
        exposed[:, 1:]   |= infra[:, :-1]          # left
        exposed[:, :-1]  |= infra[:, 1:]           # right
        exposed[1:, 1:]   |= infra[:-1, :-1]       # up-left
        exposed[1:, :-1]  |= infra[:-1, 1:]        # up-right
        exposed[:-1, 1:]  |= infra[1:, :-1]        # down-left
        exposed[:-1, :-1] |= infra[1:, 1:]         # down-right
        m[(m == CORE) & exposed] = EDGE

    def _analyse(self):
        core = int(np.count_nonzero(self.forest_map == CORE))
        edge = int(np.count_nonzero(self.forest_map == EDGE))
        infra = GRID_SIZE * GRID_SIZE - core - edge

        carbon_c = (core * CARBON_DENSITY_CORE
                    + edge * CARBON_DENSITY_CORE * (1.0 - EDGE_DEGRADATION)) * HA_PER_CELL
        co2e = carbon_c * C_TO_CO2
        value = co2e * IDR_PER_TCO2E

        total_core, largest, fragments, score = self._connectivity()
        return {
            "core": core, "edge": edge, "infra": infra,
            "co2e": co2e, "value": value,
            "largest": largest, "fragments": fragments, "score": score,
        }

    def _connectivity(self):
        seen = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
        sizes = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.forest_map[r][c] == CORE and not seen[r][c]:
                    size = 0
                    stack = [(r, c)]
                    seen[r][c] = True
                    while stack:
                        y, x = stack.pop()
                        size += 1
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE
                                    and self.forest_map[ny][nx] == CORE and not seen[ny][nx]):
                                seen[ny][nx] = True
                                stack.append((ny, nx))
                    sizes.append(size)
        total_core = sum(sizes)
        largest = max(sizes) if sizes else 0
        fragments = len(sizes)
        score = (largest / total_core * 100.0) if total_core else 0.0
        return total_core, largest, fragments, score

    # ------------------------------------------------------------- SIMULATE
    def simulate(self):
        self.compute_edges()
        self.render_grid()
        self._update_live_readout()
        stats = self._analyse()
        self.history_core_pct.append(stats["core"] / (GRID_SIZE * GRID_SIZE) * 100.0)

        level, message = self.show_dashboard(stats=stats)
        if level == "critical":
            Alert.show(self.root, "critical", "Ecosystem Critical", message)
        elif level == "warning":
            Alert.show(self.root, "warning", "Habitat Warning", message)

    # ======================================================================
    # VIEW SWITCHING (Section 2 <-> Section 3)
    # ======================================================================
    def show_workspace(self):
        # Full-width editing view. We never place_forget the workspace itself:
        # keeping it mapped means the grid canvas keeps painting and never
        # needs a stray click to reappear when returning from the results view.
        self.dashboard_frame.place_forget()
        self.workspace_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

    def show_dashboard(self, stats=None):
        if stats is None:
            stats = self._analyse()
        # Split view: the editor/grid stays on the left (just narrower) and the
        # results dock on the right — the workspace is only resized, never
        # unmapped, so the grid remains visible throughout.
        self.workspace_frame.place(relx=0, rely=0, relwidth=0.57, relheight=1)
        self.dashboard_frame.place(relx=0.57, rely=0, relwidth=0.43, relheight=1)
        # Settle geometry before measuring canvas sizes, else the first paint
        # reads stale 1x1 dimensions and charts render blank.
        self.dashboard_frame.update_idletasks()
        result = self._update_dashboard(stats)
        # Safety-net redraw next idle cycle, in case we raced the window manager.
        self.dashboard_frame.after(60, self._redraw_all_charts)
        return result

    # ======================================================================
    # SECTION 3 — ECOLOGICAL DASHBOARD
    # ======================================================================
    def _build_dashboard(self):
        self.dashboard_frame = tk.Frame(self.content, bg=BG_PANEL)

        top = tk.Frame(self.dashboard_frame, bg=BG_PANEL)
        top.pack(fill="x", padx=24, pady=(20, 4))
        tk.Label(top, text="ECOLOGICAL DASHBOARD", font=(UI_FONT, 14, "bold"),
                 fg=FG_TEXT, bg=BG_PANEL).pack(side="left")
        round_button(top, "Re-Simulate", self.show_workspace, ACCENT_DK, ACCENT,
                    fg="#ffffff", bold=True, size=10, height=36, icon="reset",
                    bg=BG_PANEL).pack(side="right")

        body = tk.Frame(self.dashboard_frame, bg=BG_PANEL)
        body.pack(fill="both", expand=True, padx=24, pady=(10, 20))
        body.grid_columnconfigure(0, weight=3, uniform="d")
        body.grid_columnconfigure(1, weight=2, uniform="d")
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=BG_PANEL)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right = tk.Frame(body, bg=BG_PANEL)
        right.grid(row=0, column=1, sticky="nsew")

        # ---- stat bento grid ----
        grid = tk.Frame(left, bg=BG_PANEL)
        grid.pack(fill="x")
        grid.grid_columnconfigure(0, weight=1, uniform="m")
        grid.grid_columnconfigure(1, weight=1, uniform="m")
        self._stat_card(grid, "core", "INNER FOREST", "ha", "forest", GREEN).grid(
            row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        self._stat_card(grid, "edge", "EDGE FOREST", "ha", "warning", AMBER).grid(
            row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
        self._stat_card(grid, "infra", "DEFORESTED", "ha", "building", RED).grid(
            row=1, column=0, sticky="nsew", padx=(0, 5), pady=5)
        self._stat_card(grid, "carbon", "CARBON STOCK", "t CO2e", "cloud", ACCENT).grid(
            row=1, column=1, sticky="nsew", padx=(5, 0), pady=5)

        VAL_BG = "#132a1e"
        val = self._card(left, bg=VAL_BG, border=GREEN_DK)
        val.pack(fill="x", pady=(8, 0))
        tk.Frame(val, bg=GREEN, width=4).pack(side="left", fill="y")
        icon_cv = tk.Canvas(val, width=48, height=48, bg=VAL_BG, highlightthickness=0)
        icon_cv.pack(side="left", padx=(12, 0))
        draw_icon(icon_cv, "leaf", 24, 24, 16, "#5fbd85", weight=3)
        vb = tk.Frame(val, bg=VAL_BG)
        vb.pack(side="left", fill="both", expand=True, padx=15, pady=12)
        tk.Label(vb, text="ESTIMATED CARBON VALUE", fg="#7fae92", bg=VAL_BG,
                 font=(UI_FONT, 8, "bold")).pack(anchor="w")
        self.metric_vars["value"] = tk.StringVar(value="—")
        tk.Label(vb, textvariable=self.metric_vars["value"], fg="#5fbd85", bg=VAL_BG,
                 font=(UI_FONT, 22, "bold")).pack(anchor="w", pady=(1, 1))
        tk.Label(vb, text="IDXCarbon rate · REDD+ Result-Based Payments",
                 fg="#5f8f74", bg=VAL_BG, font=(UI_FONT, 8)).pack(anchor="w")

        self.status_banner = tk.Label(left, text="Habitat status", bg=SLATE_DK,
                                      fg="white", font=(UI_FONT, 10, "bold"),
                                      anchor="w", justify="left", padx=14, pady=11,
                                      wraplength=300)
        self.status_banner.pack(fill="x", pady=(14, 4))
        self.conn_detail = tk.StringVar(value="")
        tk.Label(left, textvariable=self.conn_detail, fg=FG_MUTED, bg=BG_PANEL,
                 font=(UI_FONT, 9)).pack(anchor="w")

        # ---- trend chart (matplotlib) ----
        charts_row = tk.Frame(left, bg=BG_PANEL)
        charts_row.pack(fill="both", expand=True, pady=(14, 0))
        self.trend_fig, self.trend_ax, w = self._mpl_fig(charts_row, BG_PANEL)
        w.pack(fill="both", expand=True)

        # ---- right column: composition donut + connectivity gauge (matplotlib) ----
        donut_card = self._card(right)
        donut_card.pack(fill="both", expand=True)
        self._card_header(donut_card, "dashboard", "LAND COMPOSITION")
        self.donut_fig, self.donut_ax, w = self._mpl_fig(donut_card, BG_CARD)
        w.pack(fill="both", expand=True, padx=8, pady=8)

        gauge_card = self._card(right)
        gauge_card.pack(fill="both", expand=True, pady=(12, 0))
        self._card_header(gauge_card, "link", "HABITAT CONNECTIVITY")
        self.gauge_fig, self.gauge_ax, w = self._mpl_fig(gauge_card, BG_CARD)
        w.pack(fill="both", expand=True, padx=8, pady=8)

    def _stat_card(self, parent, key, caption, unit, icon, accent):
        # caption / big value / unit stacked vertically so a long value (e.g. a
        # six-figure carbon stock) can never push the unit off the card edge.
        card = self._card(parent)
        tk.Frame(card, bg=accent, width=4).pack(side="left", fill="y")
        icon_cv = tk.Canvas(card, width=34, height=34, bg=BG_CARD, highlightthickness=0)
        icon_cv.pack(side="left", padx=(10, 0))
        draw_icon(icon_cv, icon, 17, 17, 11, accent, weight=2)
        body = tk.Frame(card, bg=BG_CARD)
        body.pack(side="left", fill="both", expand=True, padx=(6, 10), pady=(8, 9))
        tk.Label(body, text=caption, fg=FG_MUTED, bg=BG_CARD,
                 font=(UI_FONT, 8, "bold"), anchor="w").pack(anchor="w")
        var = tk.StringVar(value="—")
        self.metric_vars[key] = var
        tk.Label(body, textvariable=var, fg=FG_TEXT, bg=BG_CARD,
                 font=(UI_FONT, 15, "bold"), anchor="w").pack(anchor="w")
        tk.Label(body, text=unit, fg=FG_FAINT, bg=BG_CARD,
                 font=(UI_FONT, 8), anchor="w").pack(anchor="w")
        return card

    def _card_header(self, parent, icon, text):
        """Small icon + caption header used on the dashboard chart cards."""
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(anchor="w", fill="x", padx=12, pady=(11, 0))
        ic = tk.Canvas(row, width=18, height=18, bg=BG_CARD, highlightthickness=0)
        ic.pack(side="left")
        draw_icon(ic, icon, 9, 9, 6, ACCENT, weight=2)
        tk.Label(row, text=text, font=(UI_FONT, 9, "bold"), fg=FG_MUTED,
                 bg=BG_CARD).pack(side="left", padx=6)
        return row

    def _update_dashboard(self, stats):
        self._last_stats = stats
        self.metric_vars["core"].set("%d" % stats["core"])
        self.metric_vars["edge"].set("%d" % stats["edge"])
        self.metric_vars["infra"].set("%d" % stats["infra"])
        self.metric_vars["carbon"].set(fmt_int(stats["co2e"]))
        self.metric_vars["value"].set(fmt_idr(stats["value"]))

        level, banner_text, colour = self._status(stats)
        icon = {"healthy": "OK", "warning": "!", "critical": "X"}[level]
        self.status_banner.config(text="%s   %s" % (icon, banner_text), bg=colour)
        self.conn_detail.set(
            "Fragments: %d      Largest block: %d ha      Connectivity: %.0f%%"
            % (stats["fragments"], stats["largest"], stats["score"]))

        self._redraw_all_charts()
        return level, banner_text

    def _status(self, stats):
        core = stats["core"]
        largest = stats["largest"]
        score = stats["score"]
        core_frac = core / (GRID_SIZE * GRID_SIZE)
        if core == 0:
            return ("critical",
                    "Inner forest eliminated. No habitat remains for endemic species.",
                    RED_DK)
        if largest < MIN_VIABLE_CORE_HA:
            return ("critical",
                    "No viable habitat block. Largest intact core is only %d ha "
                    "(below the %d ha threshold). Endemic species face genetic "
                    "isolation." % (largest, MIN_VIABLE_CORE_HA), RED_DK)
        if score < 60 or stats["fragments"] >= 4:
            return ("warning",
                    "Habitat fragmented into %d patches. Biological corridors are "
                    "breaking down." % stats["fragments"], AMBER_DK)
        if core_frac < SAFE_CORE_FRACTION:
            return ("warning",
                    "Core habitat below the %.0f%% safe threshold."
                    % (SAFE_CORE_FRACTION * 100), AMBER_DK)
        return ("healthy",
                "Inner forest is contiguous. Endemic habitat is secure.", GREEN_DK)

    # ------------------------------------------------------------ CHARTS
    def _redraw_all_charts(self):
        if self._last_stats is None:
            return
        self._draw_trend_chart(self._last_stats)
        self._draw_donut(self._last_stats)
        self._draw_gauge(self._last_stats)

    def _draw_trend_chart(self, stats):
        ax = self.trend_ax
        ax.clear()
        ax.set_facecolor("#0e1119")
        vals = self.history_core_pct
        xs = list(range(len(vals)))
        ax.fill_between(xs, vals, 0, color=ACCENT, alpha=0.15, zorder=2)
        ax.plot(xs, vals, color=ACCENT, lw=2, marker="o", ms=4,
                markerfacecolor=ACCENT, markeredgecolor=BG_PANEL, zorder=3)
        ax.axhline(SAFE_CORE_FRACTION * 100, color=RED, ls="--", lw=1)
        ax.text(0.995, SAFE_CORE_FRACTION * 100 + 2,
                "safe %d%%" % (SAFE_CORE_FRACTION * 100),
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                color=RED, fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, max(1, len(vals) - 1))
        ax.set_title("Core habitat trend  (% per simulation step)", color=FG_MUTED,
                     fontsize=8, loc="left", fontweight="bold", pad=8)
        ax.tick_params(colors=FG_FAINT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.grid(True, color="#1c2130", lw=0.5, ls=":")
        try:
            self.trend_fig.tight_layout()
        except Exception:
            pass
        self.trend_fig.canvas.draw_idle()

    def _draw_donut(self, stats):
        ax = self.donut_ax
        ax.clear()
        total = GRID_SIZE * GRID_SIZE
        core = stats["core"]
        vals = [core, stats["edge"], stats["infra"]]
        if sum(vals) == 0:
            vals = [1, 0, 0]
        ax.pie(vals, colors=[COLOR_CORE, COLOR_EDGE, COLOR_INFRA],
               startangle=90, counterclock=False,
               wedgeprops=dict(width=0.42, edgecolor=BG_CARD, linewidth=2))
        ax.text(0, 0.08, "%d%%" % round(core / total * 100), ha="center",
                va="center", color=FG_TEXT, fontsize=16, fontweight="bold")
        ax.text(0, -0.18, "core forest", ha="center", va="center",
                color=FG_MUTED, fontsize=8)
        ax.set_aspect("equal")
        self.donut_fig.canvas.draw_idle()

    def _draw_gauge(self, stats):
        ax = self.gauge_ax
        ax.clear()
        ax.axis("off")
        score = stats["score"]
        colour = GREEN if score >= 60 else (AMBER if score >= 35 else RED)
        # semicircle track + coloured fill growing from the left
        ax.add_patch(mpatches.Wedge((0, 0), 1.0, 0, 180, width=0.30,
                                    facecolor="#232838"))
        ang = 180 - 180 * (score / 100.0)
        ax.add_patch(mpatches.Wedge((0, 0), 1.0, ang, 180, width=0.30,
                                    facecolor=colour))
        ax.text(0, 0.34, "%.0f%%" % score, ha="center", va="center",
                color=FG_TEXT, fontsize=17, fontweight="bold")
        ax.text(0, 0.10, "connectivity score", ha="center", va="center",
                color=FG_MUTED, fontsize=8)
        ax.text(0, -0.16, "%d fragments  ·  largest %d ha"
                % (stats["fragments"], stats["largest"]), ha="center",
                va="center", color=FG_FAINT, fontsize=7)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.28, 1.15)
        ax.set_aspect("equal")
        self.gauge_fig.canvas.draw_idle()

    # --------------------------------------------------------------- RENDER
    def render_grid(self):
        colours = {CORE: COLOR_CORE, EDGE: COLOR_EDGE, INFRA: COLOR_INFRA}
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self.map_canvas.itemconfig(self.rects[r][c],
                                           fill=colours[int(self.forest_map[r][c])])


# ==========================================================================
# SECTION 1 — SPLASH / INTRO SCREEN
#
# A short cinematic intro: the whole scene fades in, the two logo marks
# reveal in sequence with a growing divider, the product line fades up, and
# a premium loading bar fills at the bottom — then the scene fades out into
# the app. Two PNGs from ./assets/logo/ are used when present (first
# alphabetically = primary mark, second = partner mark); otherwise a
# typographic wordmark stands in.
# ==========================================================================
class Splash:
    STATUS_MESSAGES = [
        "Initialising spatial engine…",
        "Loading edge-effect model…",
        "Calibrating carbon valuation…",
        "Mapping habitat connectivity…",
        "Ready.",
    ]

    # cinematic timeline (milliseconds)
    FADE_IN  = 420
    TEXT_AT  = 820
    BAR_AT   = 1080
    BAR_END  = 2350
    FADE_OUT = 2450
    TOTAL    = 2850

    def __init__(self, root, on_finished):
        self.root = root
        self.on_finished = on_finished
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.configure(bg=BG_APP)
        for k, v in (("-topmost", True), ("-alpha", 0.0)):
            try:
                self.win.attributes(k, v)
            except tk.TclError:
                pass

        self.root.update_idletasks()
        W = max(self.root.winfo_width(), 1200)
        H = max(self.root.winfo_height(), 800)
        x = self.root.winfo_x() if self.root.winfo_x() > 0 else \
            (self.root.winfo_screenwidth() - W) // 2
        y = self.root.winfo_y() if self.root.winfo_y() > 0 else \
            (self.root.winfo_screenheight() - H) // 2
        self.W, self.H = W, H
        self.win.geometry("%dx%d+%d+%d" % (W, H, x, y))

        self.cv = tk.Canvas(self.win, width=W, height=H, bg=BG_APP,
                            highlightthickness=0)
        self.cv.pack(fill="both", expand=True)

        self.logos = self._load_logos()
        self._t0 = None
        self._closed = False
        self._draw_backdrop()
        self._tick()

    # ---------------------------------------------------------- LOGO LOAD
    def _load_logos(self):
        logos = []
        try:
            if os.path.isdir(LOGO_DIR):
                files = sorted(f for f in os.listdir(LOGO_DIR)
                              if f.lower().endswith(".png"))
                for f in files[:2]:
                    try:
                        img = tk.PhotoImage(file=os.path.join(LOGO_DIR, f))
                        big = max(img.width(), img.height())
                        if big > 170:               # downscale large marks to fit
                            img = img.subsample(max(1, round(big / 155)))
                        logos.append(img)
                    except Exception:
                        pass
        except Exception:
            pass
        return logos

    def _draw_backdrop(self):
        """Static, subtle concentric glow behind the marks (drawn once)."""
        cx, cy = self.W / 2, self.H / 2 - 30
        for i, rr in enumerate((360, 260, 170)):
            col = _blend(BG_APP, ACCENT_SOFT, 0.55 - i * 0.14)
            self.cv.create_oval(cx - rr, cy - rr * 0.55, cx + rr, cy + rr * 0.55,
                                outline=col, width=1)

    # -------------------------------------------------------------- FRAME
    def _tick(self):
        if self._closed:
            return
        if self._t0 is None:
            self._t0 = time.time()
        e = (time.time() - self._t0) * 1000.0

        # global window alpha — fade the whole scene in, then out
        if e < self.FADE_IN:
            a = e / self.FADE_IN
        elif e < self.FADE_OUT:
            a = 1.0
        else:
            a = 1.0 - (e - self.FADE_OUT) / (self.TOTAL - self.FADE_OUT)
        try:
            self.win.attributes("-alpha", _clamp(a, 0.0, 1.0))
        except tk.TclError:
            pass

        self._draw_frame(e)
        if e >= self.TOTAL:
            self._finish()
            return
        self.win.after(24, self._tick)

    @staticmethod
    def _reveal(e, t_start, dur=320):
        return _clamp((e - t_start) / dur, 0.0, 1.0)

    def _draw_frame(self, e):
        self.cv.delete("fx")
        cx, cy = self.W / 2, self.H / 2 - 40
        gap = 132 if self.logos else 82

        # left mark reveals first, divider grows, then right mark
        self._draw_mark(cx - gap, cy, 0, self._reveal(e, 140))
        td = self._reveal(e, 380, 280)
        if td > 0:
            half = 30 * td
            self.cv.create_line(cx, cy - half, cx, cy + half,
                                fill=_blend(BG_APP, BORDER_HI, td), width=1, tags="fx")
        self._draw_mark(cx + gap, cy, 1, self._reveal(e, 470))

        tt = self._reveal(e, self.TEXT_AT, 360)
        if tt > 0:
            title_y = cy + (120 if self.logos else 64)
            self.cv.create_text(
                cx, title_y,
                text="Forest Carbon & Edge-Effect Decision Support System",
                fill=_blend(BG_APP, FG_MUTED, tt), font=(UI_FONT, 11), tags="fx")

        if e >= self.BAR_AT:
            self._draw_bar(e)

    def _draw_mark(self, mx, cy, index, t):
        if t <= 0.001:
            return
        if index < len(self.logos):
            if t >= 0.12:            # opaque logos can't alpha-blend; pop them in
                try:
                    img = self.logos[index]
                    hw, hh = img.width() / 2, img.height() / 2
                    pad = 13
                    round_rect(self.cv, mx - hw - pad, cy - hh - pad,
                               mx + hw + pad, cy + hh + pad, 16,
                               fill="#ffffff", outline="", tags="fx")
                    self.cv.create_image(mx, cy, image=img, anchor="center", tags="fx")
                    return
                except Exception:
                    pass
        name = "GIS.PY" if index == 0 else "FOREST DSS"
        target = FG_TEXT if index == 0 else FG_MUTED
        size = 30 if index == 0 else 13
        self.cv.create_text(mx, cy, text=name, fill=_blend(BG_APP, target, t),
                            font=(UI_FONT, size, "bold"), tags="fx")

    def _draw_bar(self, e):
        pct = _clamp((e - self.BAR_AT) / (self.BAR_END - self.BAR_AT), 0.0, 1.0)
        bar_w = min(440, self.W * 0.42)
        bx0 = (self.W - bar_w) / 2
        bx1 = bx0 + bar_w
        by = self.H - 100

        msg_i = min(len(self.STATUS_MESSAGES) - 1, int(pct * len(self.STATUS_MESSAGES)))
        self.cv.create_text(self.W / 2, by - 20, text=self.STATUS_MESSAGES[msg_i],
                            fill=FG_MUTED, font=(UI_FONT, 9), tags="fx")
        round_rect(self.cv, bx0, by, bx1, by + 5, 2.5, fill="#1a1e29",
                   outline="", tags="fx")
        fx = bx0 + bar_w * pct
        if fx > bx0 + 2:
            round_rect(self.cv, bx0, by, fx, by + 5, 2.5, fill=ACCENT,
                       outline="", tags="fx")
            self.cv.create_oval(fx - 4, by - 1, fx + 4, by + 6,
                                fill=_blend(ACCENT, "#ffffff", 0.35),
                                outline="", tags="fx")
        self.cv.create_text(bx1, by + 18, anchor="e", text="%d%%" % round(pct * 100),
                            fill=FG_FAINT, font=(UI_FONT, 8, "bold"), tags="fx")

    def _finish(self):
        self._closed = True
        try:
            self.win.destroy()
        except Exception:
            pass
        self.on_finished()


if __name__ == "__main__":
    root = tk.Tk()
    app = GisPyApp(root)
    root.mainloop()
