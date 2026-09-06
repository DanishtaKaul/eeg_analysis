

# -*- coding: utf-8 -*-
"""
Stitch panel PNGs into the age main-effect composite figure.

The figure is a TFR beside a single topomap:
  - "single":   one image fills the row (the TFR).
  - "topomaps": the topomap head, with a fresh t-value colourbar drawn beside it.

Panel letters come from "letter" (single) or "letters" (one per head).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image, ImageDraw, ImageFont

# ------------------ SETTINGS ------------------
FIG_WIDTH_IN = 6.8       # final figure width in inches (two-column width)
DPI = 300                # output resolution
# vertical gap between stacked panels (fraction of panel height)
HSPACE = 0.10
TOPO_GAP_PX = 24         # horizontal gap, in pixels, between topomap heads
TOPO_LABEL_FRAC = 0.10   # topomap label font size as a fraction of head height
TOPO_LABEL_GAP_PX = 10   # vertical gap between a topomap head and its label
LETTER_PAD_FRAC = 0.12   # rightward nudge of a topo letter, as a fraction of head width


def _hex2rgb(c):
    # Accept a "#rrggbb" string or an (r, g, b) tuple; return 0-255 ints.
    if isinstance(c, str):
        c = c.lstrip("#")
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(c)
# ----------------------------------------------


def load_rgb(path):
    # Read an image as a float RGB array with values in [0, 1].
    return np.asarray(Image.open(path).convert("RGB"), dtype=float) / 255.0


def autocrop(im, thr=0.985):
    # Trim the surrounding white margin. A pixel counts as content if any of its
    # channels is darker than thr; rows/columns with no content are removed.
    nonwhite = np.any(im[..., :3] < thr, axis=-1)
    if not nonwhite.any():
        return im
    rows = np.where(nonwhite.any(axis=1))[0]
    cols = np.where(nonwhite.any(axis=0))[0]
    return im[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def split_head_cbar(im, thr=0.985, min_gap=6):
    # Separate a topomap head from its colourbar. The colourbar sits in the right
    # half with a band of fully white columns between it and the head; the first
    # such gap marks the split. Returns (head, cbar); cbar is None if no gap.
    # True where a column has content
    ink_cols = np.any(im[..., :3] < thr, axis=-1).any(axis=0)
    w = len(ink_cols)
    c = w // 2  # start searching from the middle rightwards
    while c < w:
        if not ink_cols[c]:                 # found a white column
            j = c
            while j < w and not ink_cols[j]:  # measure the white run
                j += 1
            if j - c >= min_gap:            # wide enough to be the head/bar gap
                return im[:, :c], im[:, j:]
            c = j
        else:
            c += 1
    return im, None


def resize_to_height(im, target_h):
    # Scale an image to a fixed pixel height, preserving aspect ratio.
    pil = Image.fromarray((im * 255).astype(np.uint8))
    new_w = max(1, int(round(pil.width * target_h / pil.height)))
    pil = pil.resize((new_w, target_h), Image.LANCZOS)
    return np.asarray(pil, dtype=float) / 255.0


def label_strip(width, text, head_h, color=(0, 0, 0), frac=None, gap=None):
    # White strip holding one centred label (in `color`) below a topomap head.
    if frac is None:
        frac = TOPO_LABEL_FRAC
    if gap is None:
        gap = TOPO_LABEL_GAP_PX
    fontsize = max(8, int(round(head_h * frac)))
    font = ImageFont.truetype(font_manager.findfont("Arial"), fontsize)
    strip_h = fontsize + gap + 2
    canvas = Image.new("RGB", (width, strip_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    draw.text(((width - (r - l)) / 2 - l, gap - t),
              text, fill=tuple(color), font=font)
    return np.asarray(canvas, dtype=float) / 255.0


def detect_tfr_layout(im):
    # Find, as fractions of the (autocropped) TFR image width, the heatmap box
    # (left, right) and the F-Value colourbar body (left, right). Used to align
    # the topomap row and its colourbar to the TFR above it.
    H, W, _ = im.shape
    dark = (im < 0.35).all(axis=2)
    coldark = dark.sum(axis=0) / H
    spine = [x for x in range(W) if coldark[x] > 0.45]
    groups, s, prev = [], spine[0], spine[0]
    for x in spine[1:]:
        if x - prev > 5:
            groups.append((s, prev))
            s = x
        prev = x
    groups.append((s, prev))
    plot_l, plot_r = groups[0][0] / W, groups[1][1] / W
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    red = (R - np.maximum(G, B) > 0.18) & (R > 0.25)
    redcol = red.sum(axis=0) / H
    bar = redcol > 0.40
    runs, s = [], None
    for x in range(W):
        if bar[x] and s is None:
            s = x
        elif not bar[x] and s is not None:
            runs.append((s, x - 1))
            s = None
    if s is not None:
        runs.append((s, W - 1))
    cl, cr = runs[-1]
    return plot_l, plot_r, cl / W, cr / W


def compose_topo_row(topo_paths, labels=None, label_colors=None,
                     include_cbar=True, label_frac=None, label_gap=None):
    # Build a single image for a topomap row: every labelled head, optionally
    # plus one shared colourbar. Also return each head's left-edge x (pixels),
    # the total width, and the fraction of the image height taken by the head
    # circles (so an aligned colourbar can match the heads, not the labels).
    heads, cbar = [], None
    for p in topo_paths:
        head, bar = split_head_cbar(load_rgb(p))
        heads.append(autocrop(head))
        if cbar is None and bar is not None:   # keep the first colourbar found
            cbar = autocrop(bar)

    # Match all heads to a common height before joining them side by side.
    target_h = max(h.shape[0] for h in heads)
    heads = [resize_to_height(h, target_h) for h in heads]
    n_heads = len(heads)

    # Stack each head over its centred label; record the added label height.
    label_h = 0
    if labels:
        cols = label_colors or [(0, 0, 0)] * len(labels)
        labelled = []
        for head, text, col in zip(heads, labels, cols):
            strip = label_strip(head.shape[1], text, target_h, _hex2rgb(col),
                                frac=label_frac, gap=label_gap)
            label_h = strip.shape[0]
            labelled.append(np.concatenate([head, strip], axis=0))
        heads = labelled

    head_h_frac = target_h / (target_h + label_h) if label_h else 1.0

    pieces = list(heads)
    if cbar is not None and include_cbar:
        # Match the bar to head height, then pad below so it aligns with the heads only.
        bar = resize_to_height(cbar, target_h)
        if label_h:
            bar = np.concatenate(
                [bar, np.ones((label_h, bar.shape[1], 3))], axis=0)
        pieces.append(bar)

    # Concatenate the pieces left-to-right with a white spacer between them,
    # recording the left-edge x and width of every piece as it is placed.
    full_h = pieces[0].shape[0]
    gap = np.ones((full_h, TOPO_GAP_PX, 3))
    lefts, widths, x, row = [], [], 0, None
    for k, piece in enumerate(pieces):
        if k > 0:
            x += TOPO_GAP_PX
            row = np.concatenate([row, gap, piece], axis=1)
        else:
            row = piece
        lefts.append(x)
        widths.append(piece.shape[1])
        x += piece.shape[1]
    return row, lefts[:n_heads], widths[:n_heads], x, head_h_frac


def build_row_image(row):
    # Produce the final image array for one row, based on its kind. For topomaps,
    # also return each head's left-edge as a fraction of the cropped image width.
    if row["kind"] == "topomaps":
        # An align_tfr row drops the baked colourbar (a fresh one is drawn,
        # aligned to the TFR) and may use a custom label font fraction.
        include = not row.get("align_tfr")
        raw, head_left_px, head_w_px, _, head_h_frac = compose_topo_row(
            row["images"], row.get("labels"), row.get("label_colors"),
            include_cbar=include, label_frac=row.get("label_frac"),
            label_gap=row.get("label_gap"))
        # autocrop while tracking the left trim, so head positions stay valid
        nonwhite = np.any(raw[..., :3] < 0.985, axis=-1)
        cols = np.where(nonwhite.any(axis=0))[0]
        left_trim = int(cols[0]) if len(cols) else 0
        cropped = autocrop(raw)
        cw = cropped.shape[1]
        # letter x = head left edge + a small rightward pad scaled to head width
        head_fracs = [(max(0.0, lp - left_trim) + LETTER_PAD_FRAC * ww) / cw
                      for lp, ww in zip(head_left_px, head_w_px)]
        # content_frac < 1 pads white on both sides so the heads occupy only that
        # fraction of the row width (keeps a lone topomap from filling the page).
        cf = row.get("content_frac", 1.0)
        if cf < 1.0:
            new_w = int(round(cropped.shape[1] / cf))
            left = (new_w - cropped.shape[1]) // 2
            padded = np.ones((cropped.shape[0], new_w, 3))
            padded[:, left:left + cropped.shape[1]] = cropped
            head_fracs = [(f * cw + left) / new_w for f in head_fracs]
            cropped = padded
        return cropped, head_fracs, head_h_frac
    return autocrop(load_rgb(row["images"][0])), None, None


def crop_legend_off(img):
    # Remove the bottom-most text band (a single-entry cluster legend) from a
    # TFR, keeping the x-axis label. Used when only one cluster is shown.
    dark = (img < 0.3).all(axis=2)
    ink = np.where(dark.sum(axis=1) > 2)[0]
    bands, s, prev = [], ink[0], ink[0]
    for y in ink[1:]:
        if y - prev > 10:
            bands.append((s, prev))
            s = y
        prev = y
    bands.append((s, prev))
    return autocrop(img[:bands[-1][0] - 2])


def crop_top_title(img, gap=10):
    # Remove the top-most text band (a baked plot title), keeping the plot below.
    # The title is the first dark-ink band, set off from the plot by a white gap.
    dark = (img < 0.3).all(axis=2)
    ink = np.where(dark.sum(axis=1) > 2)[0]
    if ink.size == 0:
        return img
    bands, s, prev = [], ink[0], ink[0]
    for y in ink[1:]:
        if y - prev > gap:
            bands.append((s, prev))
            s = y
        prev = y
    bands.append((s, prev))
    if len(bands) < 2:
        return img
    return autocrop(img[bands[0][1] + 2:])


def build_figure_side_by_side(spec, output_dir):
    # Lay a single TFR (left) next to a single topomap (right): used when a
    # condition has one cluster, so the head sits beside the TFR, not below it.
    rows = spec["rows"]
    tfr_row = next(r for r in rows if r["kind"] == "single")
    topo_row = next(r for r in rows if r["kind"] == "topomaps")

    tfr = autocrop(load_rgb(tfr_row["images"][0]))
    if tfr_row.get("crop_legend"):
        tfr = crop_legend_off(tfr)
    if tfr_row.get("crop_title"):
        tfr = crop_top_title(tfr)
    tfr_ar = tfr.shape[1] / tfr.shape[0]
    raw, _, _, _, hhf = compose_topo_row(
        topo_row["images"], topo_row.get(
            "labels"), topo_row.get("label_colors"),
        include_cbar=False, label_frac=topo_row.get("label_frac"),
        label_gap=topo_row.get("label_gap"))
    head = autocrop(raw)
    if topo_row.get("crop_title"):
        head = crop_top_title(head)
    head_ar = head.shape[1] / head.shape[0]

    Hf = spec.get("height_in", 3.0)                 # figure height (inches)
    tfr_w = Hf * tfr_ar                             # TFR fills the full height
    gap1 = spec.get("tfr_gap", 0.10)               # space between TFR and head
    gap2, cbar_w, cbar_lab = 0.08, 0.16, 0.48
    head_h = spec.get("head_frac", 0.74) * Hf
    head_w = head_h * head_ar
    W = tfr_w + gap1 + head_w + gap2 + cbar_w + cbar_lab

    fig = plt.figure(figsize=(W, Hf))
    fx, fy = (lambda x: x / W), (lambda y: y / Hf)

    axA = fig.add_axes([0, 0, fx(tfr_w), 1.0])
    axA.imshow(tfr)
    axA.axis("off")
    axA.text(0.0, 0.99, tfr_row["letter"], transform=axA.transAxes,
             fontsize=14, fontweight="bold", va="top", ha="left")

    hx, hy = tfr_w + gap1, (Hf - head_h) / 2
    axB = fig.add_axes([fx(hx), fy(hy), fx(head_w), fy(head_h)])
    axB.imshow(head)
    axB.axis("off")
    # Letter nudged toward the head's upper-left arc (square image leaves the
    # corner empty). Tune via "letter_xy" on the topomap row if needed.
    lx, ly = topo_row.get("letter_xy", (0.06, 0.90))
    axB.text(lx, ly, topo_row.get("letters", ["B"])[0], transform=axB.transAxes,
             fontsize=14, fontweight="bold", va="top", ha="left")

    cs = topo_row.get("cbar", {})
    cx = hx + head_w + gap2
    cb_h = hhf * head_h
    cax = fig.add_axes([fx(cx), fy(hy + head_h - cb_h), fx(cbar_w), fy(cb_h)])
    sm = ScalarMappable(norm=Normalize(cs.get("vmin", 0), cs.get("vmax", 10)),
                        cmap=cs.get("cmap", "Reds"))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks(cs.get("ticks", [0, 2, 4, 6, 8, 10]))
    cb.set_label(cs.get("label", "F-Value"),
                 fontsize=cs.get("label_fontsize", 10))
    cb.ax.tick_params(labelsize=cs.get("tick_fontsize", 9.5))

    out_path = Path(output_dir) / spec["out"]
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[SAVED] {out_path}  ({round(W * 25.4)} x {round(Hf * 25.4)} mm)")


def build_figure(spec, output_dir):
    # Assemble one figure: stack its rows, add panel letters, and save.
    if spec.get("layout") == "side_by_side":
        return build_figure_side_by_side(spec, output_dir)
    rows = spec["rows"]
    built = [build_row_image(r) for r in rows]
    images = [b[0] for b in built]
    head_fracs_list = [b[1] for b in built]
    head_hfrac_list = [b[2] for b in built]

    # If any topomap row is set to align to the TFR, detect that TFR's heatmap
    # box and colourbar position (in figure coords; the TFR is drawn left=0.04).
    tfr_layout = None
    if any(r["kind"] == "topomaps" and r.get("align_tfr") for r in rows):
        tfr_idx = next(i for i, r in enumerate(rows) if r["kind"] == "single")
        pl, pr, cl, cr = detect_tfr_layout(images[tfr_idx])
        def fm(v): return 0.04 + 0.96 * v
        tfr_layout = (fm(pl), fm(pr), fm(cl), fm(cr))

    def disp_w(i, r):
        # An aligned topomap row is only as wide as the heatmap above it.
        if r["kind"] == "topomaps" and r.get("align_tfr") and tfr_layout:
            return (tfr_layout[1] - tfr_layout[0]) * FIG_WIDTH_IN
        return FIG_WIDTH_IN

    row_h = [disp_w(i, r) * (images[i].shape[0] / images[i].shape[1])
             for i, r in enumerate(rows)]
    fig = plt.figure(figsize=(FIG_WIDTH_IN, sum(row_h)))

    subs = np.atleast_1d(fig.subfigures(len(images), 1,
                                        height_ratios=row_h, hspace=HSPACE))

    for sub, im, fracs, hhf, r in zip(subs, images, head_fracs_list,
                                      head_hfrac_list, rows):
        if r["kind"] == "topomaps" and r.get("align_tfr") and tfr_layout:
            pl_f, pr_f, cl_f, cr_f = tfr_layout
            ax = sub.add_axes([pl_f, 0.0, pr_f - pl_f, 1.0])
            ax.imshow(im)
            ax.axis("off")
            for letter, frac in zip(r.get("letters", []), fracs or []):
                ax.text(frac, 0.99, letter, transform=ax.transAxes,
                        fontsize=14, fontweight="bold", va="top", ha="left")
            # Fresh F-Value colourbar, same x as the TFR colourbar, big text.
            # Span the head-circle band (top hhf of the row) so it is not squished.
            cs = r.get("cbar", {})
            cb_y0 = (1.0 - hhf) + 0.02
            cb_h = hhf - 0.06
            cax = sub.add_axes([cl_f, cb_y0, cr_f - cl_f, cb_h])
            sm = ScalarMappable(norm=Normalize(cs.get("vmin", 0), cs.get("vmax", 10)),
                                cmap=cs.get("cmap", "Reds"))
            cb = fig.colorbar(sm, cax=cax)
            cb.set_ticks(cs.get("ticks", [0, 2, 4, 6, 8, 10]))
            cb.set_label(cs.get("label", "F-Value"),
                         fontsize=cs.get("label_fontsize", 10))
            cb.ax.tick_params(labelsize=cs.get("tick_fontsize", 9.5))
        else:
            ax = sub.subplots()
            sub.subplots_adjust(left=0.04, right=1.0, top=1.0, bottom=0.0)
            ax.imshow(im)
            ax.axis("off")
            letters = r.get("letters")
            if r["kind"] == "topomaps" and letters and fracs:
                for letter, frac in zip(letters, fracs):
                    ax.text(frac, 0.99, letter, transform=ax.transAxes,
                            fontsize=14, fontweight="bold", va="top", ha="left")
            else:
                sub.text(0.002, 0.99, r["letter"], fontsize=14,
                         fontweight="bold", va="top", ha="left")

    out_path = Path(output_dir) / spec["out"]
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[SAVED] {out_path}  ({round(sum(row_h) * 25.4)} mm tall)")


# ====================================================================
# CONFIG: one entry per block, each with an output name and its rows
# ====================================================================
# composites land here (folder auto-created)
OUTPUT_DIR = Path(r"D:\Cluster_plots_last_prep\composites")


FIGURES = [
    {
        "out": "age_PREP_fig1_TFR_topo.png",   # Figure 4
        "layout": "side_by_side",
        "height_in": 3.8, "head_frac": 0.66, "tfr_gap": 0.45,
        "rows": [
            # Panel A: the age PREP TFR
            {"letter": "A", "kind": "single", "crop_title": True,
             "images": [r"D:\Cluster_plots_last_prep\cluster_outputs\age_main_PREP_ttest_last_prep1000perm_seed42__PREP__age_main_PREP_ttest_last_prep1000perm_seed42\cluster08_TFR.png"]},
            # Panel B: the single topomap, right of the TFR. t-value scale, drawn
            # fresh as a symmetric RdBu_r bar matched to the head (+/-3.63).
            {"letters": ["B"], "kind": "topomaps", "crop_title": True,
             "cbar": {"cmap": "RdBu_r", "vmin": -3.63, "vmax": 3.63, "label": "t-value",
                      "ticks": [-3, -2, -1, 0, 1, 2, 3],
                      "label_fontsize": 9, "tick_fontsize": 8.5},
             "images": [r"D:\Cluster_plots_last_prep\cluster_outputs\age_main_PREP_ttest_last_prep1000perm_seed42__PREP__age_main_PREP_ttest_last_prep1000perm_seed42\cluster08_topomap_agg.png"]},
        ],
    },
]


def main():
    # make the output folder if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in FIGURES:
        build_figure(spec, OUTPUT_DIR)


if __name__ == "__main__":
    main()
