import numpy as np
import nibabel as nib
from pathlib import Path
from cellpose import models
import itertools
import time
import json

# === config ===
INPUT_PATCH = "sub-AS9F7_patch00_crop128.nii.gz"
OUT_DIR = Path("cellpose_sweep_sub-AS9F7_patch00")
OUT_DIR.mkdir(exist_ok=True)

# parameter grid
DIAMETERS = [6, 8, 10, 12, 15]
CELLPROB_THRESHOLDS = [-4, -2, 0]
FLOW_THRESHOLDS = [0.4, 0.6]
MODELS = ["cyto3", "nuclei"]
MIN_SIZE = 15

# === load ===
nii = nib.load(INPUT_PATCH)
raw = nii.get_fdata().squeeze().astype(np.float32)
print(f"loaded {INPUT_PATCH}: shape {raw.shape}, range [{raw.min():.0f}, {raw.max():.0f}]")

# === sweep ===
results = []
combos = list(itertools.product(MODELS, DIAMETERS, CELLPROB_THRESHOLDS, FLOW_THRESHOLDS))
print(f"\nrunning {len(combos)} parameter combinations...")

for i, (model_type, diam, cellprob, flow) in enumerate(combos):
    name = f"m{model_type}_d{diam}_cp{cellprob}_fl{flow}"
    out_path = OUT_DIR / f"{name}.nii.gz"
    if out_path.exists():
        print(f"[{i+1}/{len(combos)}] {name} — already exists, skipping")
        continue
    
    print(f"\n[{i+1}/{len(combos)}] {name}")
    t0 = time.time()
    
    try:
        model = models.CellposeModel(model_type=model_type, gpu=True)
        masks, flows, _ = model.eval(
            raw,
            channels=[0, 0],
            diameter=diam,
            cellprob_threshold=cellprob,
            flow_threshold=flow,
            do_3D=True,
            z_axis=0,
            channel_axis=None,
            min_size=MIN_SIZE,
        )
        
        elapsed = time.time() - t0
        n_objects = int(masks.max())
        n_voxels = int((masks > 0).sum())
        pct = 100 * n_voxels / masks.size
        
        # save as binary mask (for consistency with other gold files)
        nib.save(nib.Nifti1Image((masks > 0).astype(np.uint8), nii.affine, header=nii.header), str(out_path))
        
        results.append({
            "name": name, "model": model_type, "diameter": diam,
            "cellprob": cellprob, "flow": flow,
            "n_objects": n_objects, "n_voxels": n_voxels, "pct": pct,
            "time_sec": elapsed
        })
        print(f"  objects={n_objects}, voxels={n_voxels} ({pct:.2f}%), time={elapsed:.1f}s")
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append({"name": name, "error": str(e)})

# save summary
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n=== SUMMARY ===")
print(f"{'name':<40} {'objects':>8} {'%vol':>6} {'time':>6}")
for r in results:
    if "error" in r:
        print(f"{r['name']:<40} FAILED")
    else:
        print(f"{r['name']:<40} {r['n_objects']:>8} {r['pct']:>5.2f}% {r['time_sec']:>5.1f}s")
