# %% [markdown]
# # Registration of MRI to MNI 152 template
# 
# ## ============================================================================
# ## SECTION 1: Setup and Data Import
# ## ============================================================================

# %%
## ============================================================================
## SECTION 1.1: Import Libraries
## ============================================================================
# All libraries are used:
# - numpy, pandas: data manipulation
# - os, fnmatch: file system operations
# - ants: neuroimaging operations (ANTsPy)
# - multiprocessing: parallel processing
# - time: timing operations
# - nibabel: NIfTI file I/O
import numpy as np 
import pandas as pd 
import os, fnmatch
import ants
from multiprocessing import Pool, cpu_count
import time
import nibabel as nib

# %%
## ============================================================================
## SECTION 1.2: Import Additional Libraries
## ============================================================================
# Additional libraries:
# - pickle: serialization (if needed)
# - glob: file pattern matching
# - matplotlib.pyplot: visualization (optional, exploration)
# - datetime: timestamp operations
# - sys: system operations
import pickle
import glob
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# %%
## OPTIONAL: Display options (not needed for pipeline)
pd.set_option('display.max_columns', 30)

# %%
## ============================================================================
## SECTION 1.3: Load Metadata
## ============================================================================
# Load metadata and PET-MRI pairs
# Old file reference (commented out): 'metafile_ADDLpipeline_abeta_mri_23_11_2023.csv'
meta = pd.read_csv('/csc/epitkane/home/atagmazi/ADDL_pipeline/scripts/metafile_completing/metafile_completed_ADNI_A4_processed_02_06_2024_shuffled.csv')

pet_mri_pairs = pd.read_csv('/csc/epitkane/home/atagmazi/ADDL_pipeline/scripts/pet_mri_pairs.csv',header=[0], index_col=[0])
pet_mri_pairs.reset_index(drop=True, inplace = True)

# %%
## ============================================================================
## SECTION 2: Filter MRI Metadata
## ============================================================================
# Filter metadata to only include MRI scans that are in PET-MRI pairs
# Old code (commented out): filter all MRI scans
meta_mri = meta[meta['Image.Data.ID'].isin(pet_mri_pairs['MRI_ID'])].copy()
#meta_mri = meta[meta['Modality'] == 'MRI']
meta_mri.reset_index(drop=True, inplace = True)

# %%
## ============================================================================
## SECTION 3: Load MNI Template
## ============================================================================
# Load MNI152 T1 template for registration
# Old template (commented out): 'mni_icbm152_t1_tal_nlin_sym_09a.nii'
mni_t1 = ants.image_read('/csc/epitkane/home/atagmazi/tpl-MNI152NLin6Asym_res-01_T1w.nii.gz')
#brain_mask = ants.image_read('/csc/epitkane/home/atagmazi/Downloads/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii')


# %%
## EXPLORATION CODE: Display mni_t1 image object (not needed for pipeline)
# mni_t1

# %%
## EXPLORATION CODE: Check template dimensions (not needed for pipeline)
# mni_t1.shape

# %%
## ============================================================================
## SECTION 4: Define Registration Function
## ============================================================================
# Register MRI scans to MNI152 template using affine transformation
# For each MRI scan:
#   1. Resample to 1mm isotropic
#   2. Register to MNI template (affine)
#   3. Apply transformation to original image
#   4. Check for NaN values
#   5. Save registered image
def registration(mri_table):
    failed_toregister = []  # To store IDs of failed registrations
    isnan = []  # To store IDs of images with NaN values

    for i in range(mri_table.shape[0]):
        image_id = mri_table['Image.Data.ID'].values[i]
        print(f"Processing sample {image_id}")

        try:
            # Read the MRI image
            mri = ants.image_read(mri_table['PATH'].values[i])

            # Resample the image
            mri_resampled = ants.resample_image(mri, (1, 1, 1), use_voxels=False, interp_type=0)

            # Perform registration
            mri_to_mni = ants.registration(
                fixed=mni_t1,
                moving=mri_resampled,
                type_of_transform='Affine',
                outprefix=f"{image_id}_mritomni"
            )

            # Get forward transformation list
            transf_list = mri_to_mni['fwdtransforms']

            # Apply the transformation
            wrap_mri = ants.apply_transforms(
                fixed=mni_t1,
                moving=mri,
                transformlist=transf_list
            )

            # Check for NaN values in the registered image
            if np.isnan(wrap_mri.numpy()).any():
                isnan.append(image_id)
                print(f"Warning: NaN values found in registered image for {image_id}")
                continue

            # Save the registered image
            output_path = f"/csc/epitkane/data/ADNI_A4/ADNI_16_04_22_A4_25_10_23_registered_mri/{image_id}_registered.nii"
            ants.image_write(wrap_mri, output_path)

            # Verify if the file was saved successfully
            if not os.path.isfile(output_path):
                failed_toregister.append(image_id)
                print(f"Error: Failed to save registered image for {image_id}")
                continue

            print(f"Sample {image_id} processed successfully.")

        except Exception as e:
            # Log any errors during processing
            failed_toregister.append(image_id)
            print(f"Error processing {image_id}: {e}")
            continue

        finally:
            # Remove temporary files from the /tmp directory
            try:
                directory = '/csc/epitkane/home/atagmazi/ADDL_pipeline/scripts/registration/'
                pattern = f"{image_id}*"

                for filename in fnmatch.filter(os.listdir(directory), pattern):
                    file_path = os.path.join(directory, filename)
                    os.remove(file_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not remove temporary files for {image_id}: {cleanup_error}")

    # Summary of failed registrations and NaN issues
    if failed_toregister:
        print(f"Registration failed for {len(failed_toregister)} images: {failed_toregister}")
    if isnan:
        print(f"Images with NaN values: {isnan}")


# %%

def split_data(data, chunk_size):
    """Splits data into chunks of size `chunk_size`."""
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

if __name__ == '__main__':
    
    chunk_size = 1000  # Define the size of each chunk

    # Split the data
    splitted_data = split_data(meta_mri, chunk_size)

    # Start multiprocessing
    start_time = time.perf_counter()

    with Pool(processes=10) as pool:
        try:
            # Map the function across the data chunks
            results = pool.map(registration, splitted_data)
        except Exception as e:
            print(f"Error during multiprocessing: {e}")
            results = []  # Set to empty list if errors occur
        finally:
            pool.close()
            pool.join()  # Ensure all processes terminate properly

    # End timing
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time:.2f} seconds - using multiprocessing")
    print("---")


# %%
sys.exit()


