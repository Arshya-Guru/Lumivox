"""Build the two main training manifests:
1. Abeta 50k patches across all datasets
2. Iba1 50k patches across ki3 datasets only

Region distribution: 40% cortex, 30% hippocampus, 20% striatum, 10% cerebellum
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lumivox.data.manifest import build_patch_manifest, save_manifest

REGION_GROUPS = {
    "cortex": (["L_Isocortex", "R_Isocortex"], 0.40),
    "hippocampus": (["L_Hippocampal formation", "R_Hippocampal formation"], 0.30),
    "striatum": (["L_Striatum", "R_Striatum"], 0.20),
    "cerebellum": (["L_Cerebellum", "R_Cerebellum"], 0.10),
}

ALL_DATASETS = [
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2",
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch3",
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch1",
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch2",
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch3",
    "/nfs/trident3/lightsheet/prado/mouse_app_vaccine_batch",
]

KI3_DATASETS = [
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch1",
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch2",
    "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch3",
]

# --- Manifest 1: Abeta across all datasets ---
print("=" * 60)
print("Building Abeta manifest (all datasets, 50k patches)")
print("=" * 60)

abeta_manifest = build_patch_manifest(
    dataset_roots=ALL_DATASETS,
    stain="Abeta",
    region_groups=REGION_GROUPS,
    n_patches=50000,
    patch_size=256,
    crop_size=128,
    seed=42,
)
save_manifest(abeta_manifest, "manifests/abeta_50k_alldata.json")

# --- Manifest 2: Iba1 across ki3 datasets only ---
print()
print("=" * 60)
print("Building Iba1 manifest (ki3 datasets only, 50k patches)")
print("=" * 60)

iba1_manifest = build_patch_manifest(
    dataset_roots=KI3_DATASETS,
    stain="Iba1",
    region_groups=REGION_GROUPS,
    n_patches=50000,
    patch_size=256,
    crop_size=128,
    seed=42,
)
save_manifest(iba1_manifest, "manifests/iba1_50k_ki3.json")

print("\nDone!")
