#!/usr/bin/env python
# coding: utf-8

# # petVAE latent feature → brain region mapping
# 
# This notebook takes the **selected latent feature indices** you obtained in `petVAE_clusters_analysis.ipynb` (e.g. `[2905, 3014, ...]`) and produces:
# 
# - **Voxelwise maps** showing where each latent feature is expressed (via correlation with PET intensity across subjects)
# - **Atlas-level summaries** (top brain regions per feature)
# 
# ## Key assumption (matches your `petVAE_tool/run_model.py`)
# Your latent CSV stores, for each scan, a single long vector formed by concatenating per-slice latent means (`mu`) in slice order:
# 
# - `latent_size` = number of latent dims per slice (e.g. 64)
# - `feature_index i` maps to:
#   - `slice_index = i // latent_size`
#   - `latent_dim  = i % latent_size`
# 
# If your CSV uses a different convention, update the mapping cell below.
# 

# In[3]:


import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import ants

import matplotlib.pyplot as plt

# Optional but recommended
#!pip install nilearn seaborn
from nilearn import image, masking, plotting, datasets
import seaborn as sns

sns.set_context("talk")
plt.rcParams["figure.dpi"] = 120


# ## Inputs
# 
# Fill these paths/values to match your environment.
# 
# You need:
# - A **latent features CSV** produced by `run_model.py` (`df.to_csv(...)`)
# - A way to map `scan_id` → **PET NIfTI path** (either via a mapping CSV or a directory pattern)
# - A **brain mask** NIfTI in the same space as the PET images (recommended)
# 

# In[4]:



mni_t1 = ants.image_read('/csc/epitkane/home/atagmazi/tpl-MNI152NLin6Asym_res-01_T1w.nii.gz')

mask_img = nib.load('/csc/epitkane/home/atagmazi/Hammer_brain_atlas/Hammers_mith_atlas_n30r83_SPM5.nii.gz')
#mask_img =  ants.resample_image_to_target(bm, mni_t1, interp_type='nearestNeighbor').astype('uint32').numpy()



# In[5]:


# Our metafile
date = '26_05_25'
#path = '/csc/epitkane/home/atagmazi/ADDL_pipeline/scripts/VAE_model/' + date
path = '/csc/epitkane/home/atagmazi/ADDL_pipeline/scripts/bimodal_VAE/' + date
meta2=  pd.read_csv(path + '/metafile_shuffled_'+ date+'.csv',header=[0], index_col=[0], low_memory = False)

meta2.loc[meta2['Research.Group']=='LMCI','Research.Group'] = 'MCI'
meta2.loc[meta2['Research.Group']=='EMCI','Research.Group'] = 'MCI'
meta2['Subject.ID'] = meta2['MRI_subID'] #something happened with Subject.ID columns, there are 0 in some rows


# ## Load latent features
# 
# Your `run_model.py` writes a CSV where:
# - **columns** are `scan_id`
# - **rows** are feature indices `0..(n_features-1)`
# 
# We’ll load it into a matrix shaped `(n_scans, n_features)` for analysis.
# 

# In[6]:


df_train = pd.read_csv(path + '/latent_features_train2_'+ date+'.csv', index_col = 0)
df_validation = pd.read_csv(path + '/latent_features_validation2_'+ date+'.csv', index_col = 0)
df_test = pd.read_csv(path + '/latent_features_test2_'+ date+'.csv', index_col = 0)

df_all = pd.concat([df_train,df_validation,df_test], axis = 1)


# In[7]:


ids = []
id_colname = []
for col in df_all.columns:
    #ids.append(col.split('_')[0]) # 1modality 
    ids.append(col.split('/')[6].split('_')[0])
    id_colname.append(col.split('/')[6])
ids_unique = list(dict.fromkeys(ids)) 
df_all.columns = id_colname


# In[8]:


# concatenate features from slices to scan level
pet_combined_df = pd.DataFrame()
for i in ids_unique:
    #Find columns that start with the current ID
    columns = [col for col in df_all.columns if col.startswith(i)]
    
    # Concatenate the matching columns vertically
    concat = pd.concat([df_all[col] for col in columns], ignore_index=True)
    
    # Assign the concatenated result to a new column in the combined DataFrame
    pet_combined_df[i] = concat
    
print('combined pet features')


# In[9]:


pet_combined_df


# In[43]:


latent_df = pet_combined_df
# ---- Feature selection results (paste from your analysis notebook) ----
SELECTED_FEATURES = [2905, 3014, 3033, 3161, 3225, 3270, 3277, 7022, 7086, 7693]
#[2905, 3014, 3161, 3225, 3277, 6894, 7022, 7086, 7150]

# ---- VAE settings (must match training/inference) ----
LATENT_SIZE = 64   # mu length per slice
# latent_df: rows = feature_index, cols = scan_id
latent_df.index = latent_df.index.astype(int)

print("latent_df shape (n_features, n_scans):", latent_df.shape)
#print("example scan_ids:", latent_df.columns[:5].tolist())

# X: (n_scans, n_features)
X_lat = latent_df.T
print("X_lat shape (n_scans, n_features):", X_lat.shape)

missing = [f for f in SELECTED_FEATURES if f not in X_lat.columns]
if missing:
    raise ValueError(f"Selected feature(s) not found in latent CSV columns: {missing}")


# ## Map feature index → (slice_index, latent_dim)
# 
# This is useful for interpretation/debugging (e.g., “feature 2905 is slice 45, latent dim 25” if `LATENT_SIZE=64`).
# 

# In[44]:


feat_map = pd.DataFrame({
    "feature_index": SELECTED_FEATURES,
})
feat_map["slice_index"] = feat_map["feature_index"] // LATENT_SIZE
feat_map["latent_dim"] = feat_map["feature_index"] % LATENT_SIZE
feat_map


# In[47]:


# ordering so same latent dim will be next to each other for easier comparison 
SELECTED_FEATURES = [2905,3033, 3161, 3225, 3014,  3270, 3277, 7693, 7022, 7086]


# ## Load PET volumes for the same scans
# 
# You can provide a `scan_id → nifti_path` mapping CSV, or point to a directory and we’ll auto-match filenames.
# 
# Important:
# - All images should be in the **same space** (same shape + affine) for voxelwise correlation.
# - Use **reconstructed PET** or **original PET** consistently.
# 

# In[ ]:


'''scan_ids = X_lat.index.astype(str).tolist()

scan_to_path = {}

if SCAN_TO_NIFTI_CSV is not None:
    m = pd.read_csv(SCAN_TO_NIFTI_CSV)
    if not {"scan_id", "nifti_path"}.issubset(set(m.columns)):
        raise ValueError("Mapping CSV must contain columns: scan_id, nifti_path")
    scan_to_path = dict(zip(m["scan_id"].astype(str), m["nifti_path"].astype(str)))
else:
    # auto-index directory by regex
    pet_dir = Path(PET_NIFTI_DIR)
    if not pet_dir.exists():
        raise FileNotFoundError(f"PET_NIFTI_DIR does not exist: {pet_dir}")

    files = list(pet_dir.glob("*.nii")) + list(pet_dir.glob("*.nii.gz"))
    rx = re.compile(PET_FILENAME_REGEX)
    for f in files:
        m = rx.search(f.name)
        if m is None:
            continue
        sid = m.group("scan_id")
        scan_to_path[sid] = str(f)

# Keep only scan_ids present in both latent CSV and PET files
available = [sid for sid in scan_ids if sid in scan_to_path]
missing_scans = sorted(set(scan_ids) - set(available))

print("Scans in latent CSV:", len(scan_ids))
print("Scans with PET NIfTI:", len(available))
if missing_scans:
    print("Missing PET for first 10 scans:", missing_scans[:10])

# Subset latent matrix to scans that have PET images
X_lat2 = X_lat.loc[available]

pet_paths = [scan_to_path[sid] for sid in available]
print("Example PET path:", pet_paths[0] if pet_paths else None)

if len(pet_paths) < 5:
    raise RuntimeError("Too few PET volumes found. Provide SCAN_TO_NIFTI_CSV or fix PET_NIFTI_DIR/regex.")
'''


# In[12]:


meta2.columns


# In[29]:


pet_paths = meta2.PET_PATH_normalised


# In[30]:


pet_paths[2]


# In[31]:


X_lat2 = X_lat  #.iloc[:10]


# In[19]:


X_lat2


# ## Mask PET data into a 2D matrix
# 
# We’ll use a brain mask to extract voxel values into a matrix `Y` with shape:
# 
# - `Y.shape = (n_scans, n_voxels)`
# 
# Then, for each selected feature, compute a voxelwise correlation map across subjects.
# 

# In[20]:


from nilearn.datasets import load_mni152_brain_mask
mask_img = load_mni152_brain_mask()


# In[58]:


#pet_imgs = [nib.load(p) for p in pet_paths]
ref_img = nib.load(pet_paths[0])

# Sanity check: same shape/affine
ref_shape = ref_img.shape
ref_aff = ref_img.affine

'''for i, img in enumerate(pet_imgs, start=1):
    if img.shape != ref_shape:
        raise ValueError(f"Shape mismatch at i={i}: {img.shape} != {ref_shape}")
'''

# Resample mask to PET space if needed
if mask_img.shape != ref_shape or not np.allclose(mask_img.affine, ref_aff):
    mask_img = image.resample_to_img(mask_img, ref_img, interpolation="nearest")

# Stack into 4D and apply mask
#pet_4d = image.concat_imgs([image.new_img_like(ref_img, img.get_fdata()) for img in pet_imgs])

# Applying brain mask
#Y = masking.apply_mask(pet_4d, mask_img)  # (n_scans, n_voxels)

Y_list = []

for p in pet_paths:
    img = nib.load(p)
    data = masking.apply_mask(img, mask_img)
    Y_list.append(data)
    del img

Y = np.vstack(Y_list)
print("Y shape (n_scans, n_voxels):", Y.shape)
#Each row = one PET scan
#Each column = one voxel inside the mask


# In[57]:


'''corr_maps = {}  # feature_index -> Nifti1Image
corr_vectors = {}  # feature_index -> (n_voxels,)

for p in pet_paths:
    img = nib.load(p)
    data = masking.apply_mask(img, mask_img)
    print(data.shape)
    
    for f in SELECTED_FEATURES:
        vals = X_lat2[f].to_numpy()
        r = corr_feature_to_voxels(vals, Yz)
        corr_vectors[f] = r
        corr_maps[f] = masking.unmask(r, mask_img)
    
    del img'''


# In[24]:


pet_4d.shape


# ## Compute voxelwise correlation maps for selected features
# 
# We compute Pearson correlation per voxel:
# 
# \[
#  r_v = \mathrm{corr}(\text{feature},\; \text{PET}_v)
# \]
# 
# Implementation uses z-scoring and a dot product for speed.
# 

# In[ ]:


X_lat2


# In[49]:


#for each feature, which brain voxels are correlated with it?
def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    s = np.nanstd(x)
    if s == 0:
        return np.zeros_like(x)
    return x / s
#Takes a vector (e.g., one feature across subjects).
#Subtracts the mean → centers data.
#Divides by standard deviation → scales it.
#Result: mean = 0, std = 1

def zscore_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd
#Works on a matrix ( Y).
#axis=0 → operates column-wise (per voxel).
# So each voxel is standardized across scans (subjects).
#If a voxel has zero variance → std set to 1 (prevents divide-by-zero)

# z-score PET voxel matrix across scans (subjects) Standartize VOXEL VALUES across scans
Yz = zscore_2d(Y) 
#Y shape: (n_scans, n_voxels)
#Yz: same shape, but each voxel is now standardized across subjects

corr_maps = {}  # feature_index -> Nifti1Image
corr_vectors = {}  # feature_index -> (n_voxels,)

n = Yz.shape[0]

def corr_feature_to_voxels(feature_values: np.ndarray, Yz: np.ndarray) -> np.ndarray:
    fz = zscore_1d(feature_values).reshape(-1, 1)  # (n_scans, 1) # standartize across LATENT FEATURES across scans
    # Pearson r per voxel = mean(fz * Yz) when both are z-scored
    r = (fz * Yz).mean(axis=0) #For each voxel, do subjects with higher feature values also have higher voxel values?
    # numeric safety
    return np.clip(r, -1.0, 1.0)

#For each voxel, do subjects with higher feature values also have higher voxel values?
#For each feature:
#Extract values across subjects
#Compute correlation with every voxel

for f in SELECTED_FEATURES:
    vals = X_lat2[f].to_numpy()
    r = corr_feature_to_voxels(vals, Yz)
    corr_vectors[f] = r
    corr_maps[f] = masking.unmask(r, mask_img)

    

print("Computed correlation maps:", len(corr_maps))


# ## Plot voxelwise maps (glass brain + stat map)
# 
# Tip: If maps look noisy, try smoothing your PET volumes before masking (e.g. `nilearn.image.smooth_img`).
# 

# In[51]:


'''BACKGROUND_IMG = datasets.load_mni152_template()
f0 = SELECTED_FEATURES[0]
img0 = corr_maps[f0]

display = plotting.plot_glass_brain(
    img0,
    colorbar=True,
    plot_abs=False,
    title=f"Feature {f0} corr with PET (glass brain)",
)

display = plotting.plot_stat_map(
    img0,
    bg_img=BACKGROUND_IMG,
    threshold=0.10,
    display_mode="ortho",
    colorbar=True,
    title=f"Feature {f0} corr with PET (threshold=0.10)",
)'''


# In[52]:


'''BACKGROUND_IMG = datasets.load_mni152_template()
for N in range(10):
    f0 = SELECTED_FEATURES[N]
    img0 = corr_maps[f0]
    
    display = plotting.plot_stat_map(
    img0,
    bg_img=BACKGROUND_IMG,
    threshold=0.10,
    display_mode="ortho",
    colorbar=True,
    title=f"Feature {f0} corr with PET (threshold=0.10)",
)
'''


# In[42]:


SELECTED_FEATURES


# In[54]:


'''import matplotlib.pyplot as plt
from nilearn import plotting, datasets

BACKGROUND_IMG = datasets.load_mni152_template()

fig, axes = plt.subplots(4, 3, figsize=(40, 20))  # 2 rows × 5 columns
axes = axes.flatten(order='F')  # flatten to 1D for easy looping

for i in range(10):
    f0 = SELECTED_FEATURES[i]
    img0 = corr_maps[f0]

    plotting.plot_stat_map(
        img0,
        bg_img=BACKGROUND_IMG,
        threshold=0.10,
        display_mode="ortho",
        colorbar=False,  # important: avoid clutter
        title=f"Feature {f0}",
        axes=axes[i],
        figure=fig
    )
axes[-1].remove()
axes[-2].remove()
#plt.tight_layout()
plt.savefig(path + f'/features_brain_plot_{date}.png', bbox_inches='tight')
plt.savefig(path + f'/features_brain_plot_{date}.svg', format='svg', bbox_inches='tight')


plt.show()'''


# ## Atlas region summary
# 
# We summarize each correlation map into brain regions using a standard atlas. Default here is Harvard-Oxford (cortical + subcortical).
# 
# For each region we compute a score:
# 
# - `mean_abs_r = mean(|r|)` within that region (you can switch to mean signed correlation)
# 

# In[37]:


#Instead of thousands of voxels, tell me which anatomical regions matter most.
# Fetch atlas (first run downloads to nilearn cache)
atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
atlas_img = atlas.maps
atlas_labels = atlas.labels

# Resample atlas to PET space
atlas_res = image.resample_to_img(atlas_img, ref_img, interpolation="nearest")
atlas_data = atlas_res.get_fdata().astype(int)

# Mask atlas to the same brain mask voxels
atlas_in_mask = masking.apply_mask(atlas_res, mask_img).astype(int)  # (n_voxels,)

# Region ids present (exclude 0 which is 'Background')
region_ids = np.unique(atlas_in_mask)
region_ids = region_ids[region_ids != 0]
print("Regions in mask:", len(region_ids))

# Precompute voxel indices per region for speed
region_to_vox = {rid: np.where(atlas_in_mask == rid)[0] for rid in region_ids}


def summarize_regions(r: np.ndarray, top_k: int = 15, use_abs: bool = True) -> pd.DataFrame:
    rows = []
    for rid, vox_idx in region_to_vox.items():
        if vox_idx.size == 0:
            continue
        vals = r[vox_idx]
        score = float(np.mean(np.abs(vals))) if use_abs else float(np.mean(vals))
        label = atlas_labels[rid] if rid < len(atlas_labels) else f"Region {rid}"
        rows.append({"region_id": int(rid), "region": label, "score": score, "n_vox": int(vox_idx.size)})
    df = pd.DataFrame(rows).sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    return df

# Example summary for first feature
region_df0 = summarize_regions(corr_vectors[f0], top_k=20, use_abs=True)
region_df0
#score - average correlation for the region


# In[69]:


r.shape


# In[38]:


plt.figure(figsize=(10, 6))
sns.barplot(data=region_df0, y="region", x="score", color="#4C72B0")
plt.title(f"Top regions for feature {f0} (mean |corr|)")
plt.xlabel("mean |r|")
plt.ylabel("")
plt.tight_layout()
plt.show()


# ## Batch export (NIfTI + CSV)
# 
# This writes:
# - one NIfTI correlation map per feature
# - one CSV of top atlas regions per feature
# 

# In[39]:


OUT_DIR = Path(path+"/feature_brain_maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

region_rows = []

for f in SELECTED_FEATURES:
    # Save NIfTI
    out_nii = OUT_DIR / f"feature_{f}_corr_map.nii.gz"
    nib.save(corr_maps[f], str(out_nii))

    # Atlas summary
    reg = summarize_regions(corr_vectors[f], top_k=25, use_abs=True)
    reg.insert(0, "feature_index", f)
    reg["slice_index"] = f // LATENT_SIZE
    reg["latent_dim"] = f % LATENT_SIZE
    region_rows.append(reg)

regions_all = pd.concat(region_rows, ignore_index=True)
regions_csv = OUT_DIR / "feature_region_summary.csv"
regions_all.to_csv(regions_csv, index=False)

print("Saved maps to:", OUT_DIR)
print("Saved region summary:", regions_csv)
regions_all.head(10)


# In[ ]:




