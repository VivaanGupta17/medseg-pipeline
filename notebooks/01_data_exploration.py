# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (medseg)
#     language: python
#     name: medseg
# ---

# %% [markdown]
# # Notebook 01: BraTS Data Exploration & Preprocessing QC
#
# **MedSeg Pipeline** | Johns Hopkins BME × Clinical AI
#
# This notebook demonstrates:
# 1. Loading and visualising BraTS 2021 multi-modal MRI volumes
# 2. Segmentation label distribution analysis
# 3. Intensity normalisation comparison (z-score vs. percentile clipping)
# 4. Data augmentation visualisation
# 5. Dataset-level statistics for FDA GMLP documentation
# 6. DICOM loading with windowing for CT data
#
# **Prerequisites:** Install dependencies (`pip install -e ..`) and
# download BraTS 2021 training data (see `README.md`).

# %% [markdown]
# ## 0. Setup & Imports

# %%
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path("..").resolve() / "src"))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import ndimage

# MedSeg Pipeline imports
from data.dicom_loader import BraTSDataset, load_nifti, apply_preset_windowing
from data.preprocessing import (
    IntensityNormalizer,
    VolumeResampler,
    MedicalAugmentation,
    PatchExtractor,
)
from utils.visualization import (
    overlay_segmentation,
    BRATS_COLORS,
    BRATS_LABELS,
)

# Configure matplotlib
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "font.family": "sans-serif",
})

print("Setup complete.")

# %% [markdown]
# ## 1. Load a BraTS Case

# %%
# ─── Configure path ─────────────────────────────────────────────────────────
DATA_ROOT = Path("../data/brats2021/train")  # Update to your path
# ─────────────────────────────────────────────────────────────────────────────

if not DATA_ROOT.exists():
    print(f"⚠  Data root not found: {DATA_ROOT}")
    print("   Update DATA_ROOT to your BraTS 2021 training data directory.")
    print("   Proceeding with synthetic data for demonstration.")
    USE_SYNTHETIC = True
else:
    USE_SYNTHETIC = False
    cases = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
    print(f"Found {len(cases)} cases in {DATA_ROOT}")
    case_dir = cases[0]
    case_id = case_dir.name
    print(f"Loading case: {case_id}")

# %%
if USE_SYNTHETIC:
    # Generate synthetic multi-modal MRI and segmentation for demonstration
    np.random.seed(42)
    H, W, D = 240, 240, 155
    
    # Synthetic brain (ellipse)
    coords = np.mgrid[:H, :W, :D]
    brain = (
        ((coords[0] - H//2) / (H//3))**2 +
        ((coords[1] - W//2) / (W//3))**2 +
        ((coords[2] - D//2) / (D//2.5))**2
    ) < 1.0
    
    # Generate 4 modalities with different contrast
    def make_modality(bias, noise_std, brain_mask):
        vol = np.random.normal(0, noise_std, (H, W, D)).astype(np.float32)
        vol[brain_mask] += bias
        return vol
    
    t1   = make_modality(600, 30, brain)
    t1ce = make_modality(700, 30, brain)
    t2   = make_modality(400, 25, brain)
    flair= make_modality(500, 35, brain)
    
    # Synthetic tumour in a corner of the brain
    tumour = np.zeros((H, W, D), dtype=np.int64)
    cx, cy, cz = H//2 + 20, W//2 + 10, D//2
    for c, (r, label) in enumerate([(25, 2), (18, 1), (12, 3)]):  # ED, NCR, ET
        shell = (
            ((coords[0] - cx) / r)**2 +
            ((coords[1] - cy) / r)**2 +
            ((coords[2] - cz) / r)**2
        ) < 1.0
        tumour[shell & brain] = label
    
    seg = tumour
    image_4ch = np.stack([t1, t1ce, t2, flair], axis=0)  # (4, H, W, D)
    
    print(f"Synthetic data generated: image shape {image_4ch.shape}, "
          f"seg unique values {np.unique(seg)}")
else:
    # Load real data
    modalities = {}
    for mod in ("t1", "t1ce", "t2", "flair"):
        path = case_dir / f"{case_id}_{mod}.nii.gz"
        data, affine, _ = load_nifti(path)
        modalities[mod] = data
    seg_path = case_dir / f"{case_id}_seg.nii.gz"
    seg, _, _ = load_nifti(seg_path)
    seg = seg.astype(np.int64)
    seg[seg == 4] = 3  # Remap ET: 4→3
    
    image_4ch = np.stack(list(modalities.values()), axis=0)  # (4, H, W, D)
    print(f"Loaded case {case_id}: shape {image_4ch.shape}")

# %% [markdown]
# ## 2. Multi-Modal MRI Visualisation

# %%
# Choose a representative axial slice (middle of the tumour)
if seg.any():
    z_coords = np.where(seg > 0)[2]
    mid_z = int(np.median(z_coords))
else:
    mid_z = image_4ch.shape[-1] // 2

modality_names = ["T1", "T1ce", "T2", "FLAIR"]
fig, axes = plt.subplots(2, 4, figsize=(16, 9))
fig.suptitle(f"Multi-Modal MRI — Axial Slice {mid_z}", fontsize=14, fontweight="bold")

for i, (mod_name, ax_top, ax_bot) in enumerate(
    zip(modality_names, axes[0], axes[1])
):
    vol = image_4ch[i, :, :, mid_z]
    
    # Top row: raw intensity
    im = ax_top.imshow(vol, cmap="gray", aspect="equal")
    ax_top.set_title(mod_name, fontsize=11)
    ax_top.axis("off")
    plt.colorbar(im, ax=ax_top, fraction=0.046, pad=0.04)
    
    # Bottom row: with segmentation overlay
    overlay = overlay_segmentation(vol, seg[:, :, mid_z])
    ax_bot.imshow(overlay, aspect="equal")
    ax_bot.set_title(f"{mod_name} + Seg", fontsize=11)
    ax_bot.axis("off")

# Legend
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=BRATS_COLORS[c][:3], label=BRATS_LABELS[c])
           for c in sorted(BRATS_LABELS.keys())]
fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.01))
plt.tight_layout()
plt.savefig("../figures/01_multimodal_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to figures/01_multimodal_overview.png")

# %% [markdown]
# ## 3. Tumour Subregion Analysis

# %%
# Volume statistics per class
print("Segmentation Label Distribution")
print("=" * 50)
total_brain_voxels = (image_4ch[0] > 0).sum()
total_voxels = image_4ch.shape[1] * image_4ch.shape[2] * image_4ch.shape[3]

label_info = {
    0: "Background",
    1: "NCR/NET (Necrosis)",
    2: "Peritumoral Edema",
    3: "Enhancing Tumour",
}

for label, name in label_info.items():
    count = (seg == label).sum()
    pct_total = 100.0 * count / total_voxels
    pct_brain  = 100.0 * count / total_brain_voxels if label > 0 else None
    print(f"  Class {label} [{name:25s}]: {count:8,d} voxels  ({pct_total:.2f}% total)", end="")
    if pct_brain:
        print(f"  ({pct_brain:.2f}% of brain)")
    else:
        print()

# Compound BraTS regions
wt = np.isin(seg, [1, 2, 3]).sum()
tc = np.isin(seg, [1, 3]).sum()
et = (seg == 3).sum()
print(f"\n  Whole Tumour (WT): {wt:8,d} voxels")
print(f"  Tumour Core  (TC): {tc:8,d} voxels")
print(f"  Enhancing T. (ET): {et:8,d} voxels")

# %%
# Axial slice tumour extent
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Tumour Extent: Axial Projections", fontsize=13, fontweight="bold")

flair_vol = image_4ch[3]  # FLAIR
for ax, (dim, title) in zip(axes, [(2, "Axial"), (1, "Coronal"), (0, "Sagittal")]):
    # Project along axis: max intensity for image, any for mask
    img_proj = flair_vol.max(axis=dim)
    seg_proj = seg.max(axis=dim).astype(int)
    
    # Create overlay
    overlay = overlay_segmentation(img_proj, seg_proj)
    ax.imshow(overlay, aspect="equal")
    ax.set_title(f"{title} MIP", fontsize=11)
    ax.axis("off")

plt.tight_layout()
plt.savefig("../figures/01_tumour_projections.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Intensity Normalisation Comparison

# %%
t1_raw = image_4ch[0, :, :, mid_z]

methods = {
    "Raw (HU / a.u.)": t1_raw,
    "Z-score (brain mask)": IntensityNormalizer("zscore", mask_background=True)(
        image_4ch[0, :, :, mid_z][..., np.newaxis]
    ).squeeze(),
    "Min-Max": IntensityNormalizer("minmax", mask_background=False)(
        image_4ch[0, :, :, mid_z][..., np.newaxis]
    ).squeeze(),
    "Percentile (0.5–99.5%)": IntensityNormalizer("percentile", percentile_range=(0.5, 99.5))(
        image_4ch[0, :, :, mid_z][..., np.newaxis]
    ).squeeze(),
}

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("Intensity Normalisation Methods — T1 Axial Slice", fontsize=13, fontweight="bold")

for i, (name, vol) in enumerate(methods.items()):
    ax_img = axes[0][i]
    ax_hist = axes[1][i]
    
    ax_img.imshow(vol, cmap="gray", aspect="equal")
    ax_img.set_title(name, fontsize=9)
    ax_img.axis("off")
    
    brain_vals = vol[vol != 0].ravel()
    ax_hist.hist(brain_vals, bins=80, color="#2196F3", alpha=0.7, edgecolor="none")
    ax_hist.set_xlabel("Intensity", fontsize=8)
    ax_hist.set_ylabel("Count", fontsize=8)
    ax_hist.tick_params(labelsize=7)
    if len(brain_vals) > 0:
        ax_hist.axvline(brain_vals.mean(), color="red", linestyle="--", linewidth=1.5,
                        label=f"μ={brain_vals.mean():.2f}")
        ax_hist.legend(fontsize=7)

plt.tight_layout()
plt.savefig("../figures/01_normalisation_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Data Augmentation Visualisation

# %%
# Show augmentation effects on a single slice
slice_img  = image_4ch[:, :, :, mid_z]   # (C, H, W)
slice_mask = seg[:, :, mid_z]             # (H, W)
sample = {"image": slice_img, "mask": slice_mask}

augmentor = MedicalAugmentation(
    p=1.0,  # always apply for demonstration
    flip_axes=(0, 1),
    rotate_range=20.0,
    scale_range=(0.85, 1.15),
    brightness_range=0.15,
    noise_std=0.03,
    elastic_alpha=20.0,
    elastic_sigma=4.0,
)

n_augmented = 5
fig, axes = plt.subplots(2, n_augmented + 1, figsize=(18, 6))
fig.suptitle("Data Augmentation Samples (FLAIR channel)", fontsize=13, fontweight="bold")

# Original
axes[0][0].imshow(slice_img[3], cmap="gray", aspect="equal")
axes[0][0].set_title("Original", fontsize=11)
axes[0][0].axis("off")
axes[1][0].imshow(overlay_segmentation(slice_img[3], slice_mask), aspect="equal")
axes[1][0].set_title("Seg", fontsize=11)
axes[1][0].axis("off")

# Augmented samples
np.random.seed(None)
for i in range(n_augmented):
    aug = augmentor({"image": slice_img.copy(), "mask": slice_mask.copy()})
    aug_img  = aug["image"][3]   # FLAIR channel
    aug_mask = aug["mask"]
    
    axes[0][i+1].imshow(aug_img, cmap="gray", aspect="equal")
    axes[0][i+1].set_title(f"Aug {i+1}", fontsize=11)
    axes[0][i+1].axis("off")
    
    axes[1][i+1].imshow(overlay_segmentation(aug_img, aug_mask), aspect="equal")
    axes[1][i+1].set_title(f"Seg {i+1}", fontsize=11)
    axes[1][i+1].axis("off")

plt.tight_layout()
plt.savefig("../figures/01_augmentation_samples.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Dataset-Level Statistics (for GMLP Documentation)

# %%
# If real dataset is available, compute statistics across all cases
# Otherwise show the structure of the analysis

def compute_dataset_statistics(data_root: Path, max_cases: int = 50) -> dict:
    """Compute statistics over a subset of cases."""
    cases = sorted([d for d in data_root.iterdir() if d.is_dir()])[:max_cases]
    
    stats = {
        "n_cases": len(cases),
        "tumour_volumes": [],        # Total tumour voxels per case
        "wt_dice_vs_size": [],
        "class_counts": {0: 0, 1: 0, 2: 0, 3: 0},
    }
    
    for case_dir in cases:
        case_id = case_dir.name
        seg_path = case_dir / f"{case_id}_seg.nii.gz"
        if not seg_path.exists():
            continue
        
        seg_data, _, header = load_nifti(seg_path)
        seg_data = seg_data.astype(np.int64)
        seg_data[seg_data == 4] = 3
        
        tumour_vol = (seg_data > 0).sum()
        stats["tumour_volumes"].append(tumour_vol)
        
        for cls in range(4):
            stats["class_counts"][cls] += (seg_data == cls).sum()
    
    stats["mean_tumour_volume"] = np.mean(stats["tumour_volumes"])
    stats["std_tumour_volume"]  = np.std(stats["tumour_volumes"])
    return stats

if not USE_SYNTHETIC and DATA_ROOT.exists():
    print("Computing dataset statistics (first 50 cases)...")
    stats = compute_dataset_statistics(DATA_ROOT, max_cases=50)
    
    print(f"\nDataset Statistics (n={stats['n_cases']} cases)")
    print(f"  Mean tumour volume: {stats['mean_tumour_volume']:.0f} ± "
          f"{stats['std_tumour_volume']:.0f} voxels")
    
    print("\n  Class distribution:")
    total = sum(stats["class_counts"].values())
    for cls, count in stats["class_counts"].items():
        print(f"    Class {cls}: {count:,d} ({100*count/total:.2f}%)")
    
    # Plot tumour size distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(stats["tumour_volumes"], bins=30, color="#4CAF50", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Tumour Volume (voxels)", fontsize=11)
    ax.set_ylabel("Number of Cases", fontsize=11)
    ax.set_title("BraTS 2021 Tumour Volume Distribution", fontsize=12, fontweight="bold")
    ax.axvline(stats["mean_tumour_volume"], color="red", linestyle="--",
               linewidth=2, label=f"Mean: {stats['mean_tumour_volume']:.0f}")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("../figures/01_tumour_volume_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("Using synthetic data — real dataset statistics not available.")
    print("Structure of statistics that would be computed:")
    print({
        "n_cases": "int",
        "mean_tumour_volume": "float (voxels)",
        "std_tumour_volume": "float",
        "class_counts": {0: "bg_count", 1: "ncr_count", 2: "ed_count", 3: "et_count"},
    })

# %% [markdown]
# ## 7. Patch Extraction for 3-D Training

# %%
# Demonstrate patch extraction
if image_4ch.ndim == 4:  # (C, H, W, D) for 3-D
    extractor = PatchExtractor(
        patch_size=(64, 64, 64),
        overlap=0.5,
        foreground_oversample_ratio=0.5,
    )
    
    # Random patch sampling
    img_patches, seg_patches = extractor.extract_random(
        image_4ch, seg, n_patches=4
    )
    
    print(f"Image patch batch shape: {img_patches.shape}")
    print(f"Seg patch batch shape:   {seg_patches.shape}")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    fig.suptitle("3-D Patch Extraction (64³ patches, FLAIR + segmentation)",
                 fontsize=12, fontweight="bold")
    
    for i in range(4):
        mid_patch = 32  # Middle slice of each 64³ patch
        
        img_patch = img_patches[i, 3, :, :, mid_patch]  # FLAIR channel
        seg_patch = seg_patches[i, :, :, mid_patch]
        
        axes[0][i].imshow(img_patch, cmap="gray", aspect="equal")
        axes[0][i].set_title(f"Patch {i+1} — FLAIR", fontsize=10)
        axes[0][i].axis("off")
        
        axes[1][i].imshow(overlay_segmentation(img_patch, seg_patch), aspect="equal")
        axes[1][i].set_title(f"Patch {i+1} — Seg", fontsize=10)
        axes[1][i].axis("off")
    
    plt.tight_layout()
    plt.savefig("../figures/01_patch_extraction.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 8. DICOM Loading Demo (CT Windowing)

# %%
# Demonstrate CT windowing presets
import numpy as np

# Generate synthetic CT HU values (brain scan simulation)
np.random.seed(42)
h, w = 256, 256
x, y = np.mgrid[-1:1:h*1j, -1:1:w*1j]

# Skull ring (bone ~700 HU), brain parenchyma (~30-40 HU), CSF (~0 HU)
skull = (x**2 + y**2 > 0.80**2) & (x**2 + y**2 < 1.0**2)
brain = x**2 + y**2 < 0.80**2
csf   = x**2 + y**2 < 0.15**2

ct_slice = np.zeros((h, w), dtype=np.float32)
ct_slice[brain]  = np.random.normal(35,  10, skull.sum() if brain.sum()==0 else brain.sum())
ct_slice[skull]  = np.random.normal(800, 100, skull.sum())
ct_slice[csf]    = np.random.normal(5,   5,   csf.sum())
ct_slice[~brain & ~skull] = -1000  # Air

# Apply different window presets
from data.dicom_loader import WINDOW_PRESETS, apply_windowing

windows = ["brain", "bone", "soft_tissue", "subdural"]
fig, axes = plt.subplots(1, len(windows) + 1, figsize=(18, 4))
fig.suptitle("CT Windowing Presets (Synthetic CT)", fontsize=13, fontweight="bold")

axes[0].imshow(np.clip(ct_slice, -100, 1000), cmap="gray", aspect="equal")
axes[0].set_title("Raw HU (clipped)", fontsize=11)
axes[0].axis("off")

for ax, preset in zip(axes[1:], windows):
    center, width = WINDOW_PRESETS[preset]
    windowed = apply_windowing(ct_slice, center, width)
    ax.imshow(windowed, cmap="gray", aspect="equal", vmin=0, vmax=1)
    ax.set_title(f"{preset.title()}\n(C={center}, W={width})", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("../figures/01_ct_windowing.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Summary
#
# This notebook demonstrated:
#
# | Section | Key Findings |
# |---------|-------------|
# | Multi-modal MRI | FLAIR and T2 best highlight edema; T1ce best for enhancing tumour |
# | Label distribution | Background >> Edema > NCR > ET (severe class imbalance) |
# | Intensity normalisation | Z-score (brain mask) produces most consistent histograms |
# | Data augmentation | Elastic deformations + random flips provide diverse training views |
# | Patch extraction | Foreground oversampling (50%) ensures tumour coverage in patches |
# | CT windowing | Brain window (C=40, W=80) appropriate for haemorrhage detection |
#
# **Next steps:**
# - Run `scripts/train.py` to train the Attention U-Net
# - See `02_model_training.py` for training curve analysis
# - See `03_evaluation.py` for post-training metric analysis and Grad-CAM visualisations

# %%
print("Notebook complete. All figures saved to ../figures/")
