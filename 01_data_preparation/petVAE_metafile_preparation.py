#!/usr/bin/env python3
"""
Finalizing metafile for ADDL pipeline
- Attaching links of according brain scans to metafile
- Selecting MRI and amyloid scans with needed processing
"""

## ============================================================================
## SECTION 1: Setup and Data Import
## ============================================================================

## ============================================================================
## SECTION 1.1: Import Libraries
## ============================================================================
# All libraries are used in the script:
# - numpy: for np.shape() function
# - pandas: for data manipulation and CSV operations
# - os, fnmatch: for file system operations and pattern matching
# - sys: for sys.exit() (optional, can be removed if not needed)
import numpy as np 
import pandas as pd 
import os, fnmatch
import sys

## ============================================================================
## SECTION 1.2: Import Metadata File
## ============================================================================
# Load the processed metadata file created by ADNI_A4_meta.R
# Old file reference (commented out): 'ADNI_A4_processed_9_27_2023.csv'
meta = pd.read_csv('ADNI_A4_processed_02_06_2024.csv', index_col=[0],low_memory=False)

## DEBUGGING CODE: Test file matching for specific Image.Data.ID (not needed for pipeline)
# fnmatch.filter(os.listdir('/csc/epitkane/data/ADNI_A4/ADNI_16_04_22_A4_25_10_23/'), '*I1321166.nii')
# fnmatch.filter(os.listdir('/csc/epitkane/data/ADNI_A4/ADNI_16_04_22_A4_25_10_23/'), '*I13724.nii')
# fnmatch.filter(os.listdir('/csc/epitkane/data/ADNI_A4/ADNI_16_04_22_A4_25_10_23/'), '*I1619403.nii')

## ============================================================================
## SECTION 2: Attach File Paths to Metadata
## ============================================================================
### NOTE: Run on cluster if the file is big

## ============================================================================
## SECTION 2.1: Match Image.Data.ID to Actual File Paths
## ============================================================================
# Loop through metadata and find corresponding .nii files in the data directory
# Creates PATH column with full file paths for each scan
# Note: Loop starts at index 1 (not 0) - check if this is intentional
for i in range(1,np.shape(meta)[0]+1):
    name = '*' + meta['Image.Data.ID'][i]+'.nii'
    filename = fnmatch.filter(os.listdir('/csc/epitkane/data/ADNI_A4/ADNI_16_04_22_A4_25_10_23/'), name)
    if len(filename)> 0:
        meta.loc[i,'PATH'] = '/csc/epitkane/data/ADNI_A4/ADNI_16_04_22_A4_25_10_23/' + filename[0]
        
    print(i)

## ============================================================================
## SECTION 3: Filter Scans with Valid File Paths
## ============================================================================
# Remove rows where PATH is empty (scans that were not downloaded)

## EXPLORATION CODE: Check if any PATH values are null (not needed for pipeline)
# meta['PATH'].isnull().values.any()

## EXPLORATION CODE: Count null PATH values (not needed for pipeline)
# meta['PATH'].isnull().sum()

## ============================================================================
## SECTION 3.1: Keep Only Rows with Valid File Paths
## ============================================================================
meta2 = meta[meta['PATH'].notna()]

## EXPLORATION CODE: Verify no null PATH values remain (not needed for pipeline)
# meta2['PATH'].isnull().values.any()

## EXPLORATION CODE: Check filtered dataset dimensions (not needed for pipeline)
# np.shape(meta2)

## ============================================================================
## SECTION 3.2: Save Intermediate Metafile with Paths
## ============================================================================
# Save metafile with PATH column attached
# This file can be reloaded if script was run on cluster (see Section 4)
meta2.to_csv('metafile_completed_ADNI_A4_processed_02_06_2024.csv')

## OPTIONAL: Exit script (useful when running on cluster)
# sys.exit("I am done")

## ============================================================================
## SECTION 4: Reload Metafile (if script was run on cluster)
## ============================================================================
# Upload metafile if part of the script was run on the cluster
# Skip this section if running locally and Section 3 was just executed
meta2 = pd.read_csv('metafile_completed_ADNI_A4_processed_02_06_2024.csv')

## EXPLORATION CODE: Check dataset dimensions (not needed for pipeline)
# meta2.shape

## DEBUGGING CODE: Check specific PATH value (not needed for pipeline)
# meta2.PATH[111]

## EXPLORATION CODE: Count values by Phase (not needed for pipeline)
# meta2.Phase.value_counts()

## ============================================================================
## SECTION 5: Filter by Modality and Tracer Type
## ============================================================================
## SECTION 5.1: Select MRI and Amyloid PET Scans Only
## Keep: All MRI scans + Amyloid PET scans (AV45, FBB, PIB tracers)
pd.set_option('display.max_rows', 20)
meta3 = meta2[(meta2.Modality == 'MRI') | 
      (meta2.modality_subtype == 'AV45')|
     (meta2.modality_subtype == 'FBB')|
     (meta2.modality_subtype == 'PIB')]

## EXPLORATION CODE: Count values by Project (not needed for pipeline)
# meta.Project.value_counts()
# meta2.Type.value_counts()
# meta2.Project.value_counts()
# meta3.Project.value_counts()

## ============================================================================
## SECTION 5.2: Filter by Specific Scan Types and Processing
## ============================================================================
# Select only T1 MRI and amyloid PET scans with needed preprocessing:
# - ADNI: T1 scans (MP*RAGE, T1, IR*SPGR) OR Co-registered, Averaged PET scans
# - A4: T1 scans OR Florbetapir/Flortaucipir PET scans
meta4 = meta3[((meta3.Project == 'ADNI')&((meta3.Description.str.contains('MP*RAGE|T1|IR*SPGR')) | 
     (meta3.Description.str.contains('Co-registered, Averaged'))))|
           (( meta3.Project == 'A4')&((meta3.Description.str.contains('T1*')) | 
     (meta3.Description.str.contains('Flor*'))))]

## EXPLORATION CODE: Count values by Project after final filtering (not needed for pipeline)
# meta4.Project.value_counts()

## EXPLORATION CODE: Count values by Modality (not needed for pipeline)
# meta4.Modality.value_counts()

## ============================================================================
## SECTION 6: Save Final Metafile for ADDL Pipeline
## ============================================================================
# Save final metafile with filtered scans (T1 MRI + amyloid PET) for ADDL pipeline
# Note: Comment references old file 'metafile_completed_ADNI_A4_processed_9_27_2023.csv'
meta4.to_csv('metafile_ADDLpipeline_abeta_mri_02_06_2024.csv')

