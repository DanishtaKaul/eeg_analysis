# -*- coding: utf-8 -*-
"""
Stitch panel PNGs into the TFR/topomap blocks used in the paper figures.

Each block is a vertical stack of rows:
  - "single":   one image fills the row (the TFR).
  - "topomaps": several topomap heads sharing one colourbar. Each head's own
                colourbar is cropped off and a single bar is kept on the right.

Panel letters come from "letter" (single) or "letters" (one per head).

"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

# ------------------ SETTINGS ------------------
FIG_WIDTH_IN = 6.8       # final figure width in inches (two-column width)
DPI = 300                # output resolution
# vertical gap between stacked panels (fraction of panel height)
HSPACE = 0.10
TOPO_GAP_PX = 24         # horizontal gap, in pixels, between topomap heads
TOPO_LABEL_FRAC = 0.08   # topomap label font size as a fraction of head height
TOPO_LABEL_GAP_PX = 10   # vertical gap between a topomap head and its label
LETTER_PAD_FRAC = 0.12   # rightward nudge of a topo letter, as a fraction of head width
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


def label_strip(width, text, head_h):
    # White strip holding one centred black label below a topomap head.
    fontsize = max(8, int(round(head_h * TOPO_LABEL_FRAC)))
    font = ImageFont.truetype(
        font_manager.findfont(font_manager.FontProperties(
            family="Arial", weight="normal")),
        fontsize)
    strip_h = fontsize + TOPO_LABEL_GAP_PX + 2
    canvas = Image.new("RGB", (width, strip_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    draw.text(((width - (r - l)) / 2 - l, TOPO_LABEL_GAP_PX - t),
              text, fill=(0, 0, 0), font=font)
    return np.asarray(canvas, dtype=float) / 255.0


def compose_topo_row(topo_paths, labels=None):
    # Build a single image for a topomap row: every labelled head plus one shared
    # colourbar. Also return each head's left-edge x (pixels) and the total width,
    # so panel letters can be placed above the correct head.
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
        labelled = []
        for head, text in zip(heads, labels):
            strip = label_strip(head.shape[1], text, target_h)
            label_h = strip.shape[0]
            labelled.append(np.concatenate([head, strip], axis=0))
        heads = labelled

    pieces = list(heads)
    if cbar is not None:
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
    return row, lefts[:n_heads], widths[:n_heads], x


def build_row_image(row):
    # Produce the final image array for one row, based on its kind. For topomaps,
    # also return each head's left-edge as a fraction of the cropped image width.
    if row["kind"] == "topomaps":
        raw, head_left_px, head_w_px, _ = compose_topo_row(
            row["images"], row.get("labels"))
        # autocrop while tracking the left trim, so head positions stay valid
        nonwhite = np.any(raw[..., :3] < 0.985, axis=-1)
        cols = np.where(nonwhite.any(axis=0))[0]
        left_trim = int(cols[0]) if len(cols) else 0
        cropped = autocrop(raw)
        cw = cropped.shape[1]
        # letter x = head left edge + a small rightward pad scaled to head width
        head_fracs = [(max(0.0, lp - left_trim) + LETTER_PAD_FRAC * ww) / cw
                      for lp, ww in zip(head_left_px, head_w_px)]
        return cropped, head_fracs
    return autocrop(load_rgb(row["images"][0])), None


def build_figure(spec, output_dir):
    # Assemble one figure: stack its rows, add panel letters, and save.
    built = [build_row_image(r) for r in spec["rows"]]
    images = [b[0] for b in built]
    head_fracs_list = [b[1] for b in built]

    # Each row's height is set by its image aspect ratio at the fixed figure width.
    row_h = [FIG_WIDTH_IN * (im.shape[0] / im.shape[1]) for im in images]
    fig = plt.figure(figsize=(FIG_WIDTH_IN, sum(row_h)))

    # One sub-figure per row; height_ratios keep their relative sizes, hspace sets the gap.
    subs = np.atleast_1d(fig.subfigures(len(images), 1,
                                        height_ratios=row_h, hspace=HSPACE))

    for sub, im, fracs, rowspec in zip(subs, images, head_fracs_list, spec["rows"]):
        ax = sub.subplots()
        # A small left margin leaves room for the panel letter beside the image.
        sub.subplots_adjust(left=0.04, right=1.0, top=1.0, bottom=0.0)
        ax.imshow(im)
        ax.axis("off")

        letters = rowspec.get("letters")
        if rowspec["kind"] == "topomaps" and letters and fracs:
            # One bold letter per head, at that head's top-left corner.
            for letter, frac in zip(letters, fracs):
                ax.text(frac, 0.99, letter, transform=ax.transAxes,
                        fontsize=22, fontweight="bold", va="top", ha="left")
        else:
            # Single panel letter in the sub-figure's left margin.
            sub.text(0.002, 0.99, rowspec["letter"], fontsize=24,
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
        "out": "light_PREP_fig1_TFR_topo.png",   # Figure 5 panels A-C
        "rows": [
            # Panel A: the combined TFR with both cluster outlines.
            {"letter": "A", "kind": "single",
             "images": [r"D:\Cluster_plots_last_prep\all_clusters_tfr\light_main_PREP_last_prep1000perm_seed42__PREP__ALL_CLUSTERS_TFR.png"]},
            # Panels B and C: the two topomaps, each its own letter, sharing one colourbar.
            {"letters": ["B", "C"], "kind": "topomaps",
             "labels": ["Cluster 1", "Cluster 2"],
             "images": [r"D:\Cluster_plots_last_prep\cluster_outputs\light_main_PREP_last_prep1000perm_seed42__PREP__LIGHT_MAIN\cluster00_topomap_agg.png",
                        r"D:\Cluster_plots_last_prep\cluster_outputs\light_main_PREP_last_prep1000perm_seed42__PREP__LIGHT_MAIN\cluster01_topomap_agg.png"]},
        ],
    },
    
    {
        "out": "light_RESET_fig1_TFR_topo.png",   # Figure 7 panels A-C
        "rows": [
            # Panel A: the combined TFR with both cluster outlines.
            {"letter": "A", "kind": "single",
             "images": [r"D:\Cluster_plots_last_prep\all_clusters_tfr\light_main_RESET_last_prep1000perm_seed42__RESET__ALL_CLUSTERS_TFR.png"]},
            # Panels B and C: the two topomaps, each its own letter, sharing one colourbar.
            # Order 24 then 00 so the labels match the TFR ranking (Cluster 1 = larger F-mass).
            {"letters": ["B", "C"], "kind": "topomaps",
             "labels": ["Cluster 1", "Cluster 2"],
             "images": [r"D:\Cluster_plots_last_prep\cluster_outputs\light_main_RESET_last_prep1000perm_seed42__RESET__LIGHT_MAIN\cluster24_topomap_agg.png",
                        r"D:\Cluster_plots_last_prep\cluster_outputs\light_main_RESET_last_prep1000perm_seed42__RESET__LIGHT_MAIN\cluster00_topomap_agg.png"]},
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
