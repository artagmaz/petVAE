
# %% [markdown]
# # MRI processing
# 
# ## ============================================================================
# ## SECTION 1: Setup and Data Import
# ## ============================================================================

# %%
## ============================================================================
## SECTION 1.1: Import Libraries
## ============================================================================
# All libraries are used in the script:
# - numpy, pandas: data manipulation
# - os, fnmatch: file system operations
# - nibabel: NIfTI file I/O
# - matplotlib.pyplot: visualization (optional, used in exploration)
# - scipy.stats.iqr: interquartile range calculation
# - joblib: parallel processing
# - logging: logging functionality
import numpy as np 
import pandas as pd 
import os, fnmatch
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import iqr

from joblib import Parallel, delayed
import logging

# %%
## OPTIONAL: Display options (not needed for pipeline)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 30)

# %%
## ============================================================================
## SECTION 1.2: Load Metadata
## ============================================================================
# Load shuffled metadata file
# Old code (commented out): create shuffled version from original
meta2 = pd.read_csv('metafile_completing/metafile_completed_ADNI_A4_processed_02_06_2024_shuffled.csv')


# %%
## EXPLORATION CODE: Count values by Modality (not needed for pipeline)
# meta2['Modality'].value_counts()

# %%
## ============================================================================
## SECTION 2: Filter by Modality
## ============================================================================
## SECTION 2.1: Select MRI and Amyloid PET Scans
# Keep: All MRI scans + AV45 PET scans
# Other tracers (FBB, PIB) commented out
pd.set_option('display.max_rows', 20)
meta3 = meta2[(meta2.Modality == 'MRI') | 
      (meta2.modality_subtype == 'AV45')]#|
#     (meta2.modality_subtype == 'FBB')|
#     (meta2.modality_subtype == 'PIB')]

# %%
## UNUSED CODE: Old filtering method (commented out, using mask-based approach instead)
# selecting only T1 and amyloid PET scans with needed pre-proccesing 
'''meta4 = meta3[((meta3.Project == 'ADNI')&((meta3.Description.str.contains('MP*RAGE|T1|IR*SPGR')) | 
     (meta3.Description.str.contains('Co-registered, Averaged'))))|
           (( meta3.Project == 'A4')&((meta3.modality_subtype.str.contains('T1')) | 
     (meta3.Description.str.contains('Flor*'))))]'''

# %%
## ============================================================================
## SECTION 2.2: Create Masks for Filtering
## ============================================================================
# Create boolean masks for PET and MRI scans with specific processing/sequences
pet_mask = ((meta3.Modality == 'PET')&(meta3.Description.str.contains('Co-registered, Averaged')|
            meta3.Description.str.contains('^Flor.*')))
mri_mask = (meta3.Modality == 'MRI')&(meta3.Description.str.contains('MP.*RAGE|T1|IR.*SPGR')|
            meta3.modality_subtype.str.contains('T1'))

# %%
## ============================================================================
## SECTION 2.3: Apply Filters
## ============================================================================
# Filter metadata using PET and MRI masks
meta4 = meta3[pet_mask|mri_mask]
meta4.reset_index(drop=True, inplace = True)

# %%
## ============================================================================
## SECTION 3: Extract Year from Study Date
## ============================================================================
# Extract year from Study.Date column for temporal analysis
meta4 = meta4.copy()
meta4['year'] = ''
for i in range(meta4.shape[0]):
    meta4['year'].values[i] = meta4['Study.Date'].values[i].split('/')[2]

# %%
## ============================================================================
## SECTION 4: Separate PET and MRI Metadata
## ============================================================================
# Split metadata into PET and MRI dataframes for pairing
pet_meta = meta4[meta4.Modality == 'PET']
pet_meta.reset_index(drop=True, inplace = True)

mri_meta = meta4[meta4.Modality == 'MRI']
mri_meta.reset_index(drop=True, inplace = True)



# %%
## ============================================================================
## SECTION 5: Create PET-MRI Pairing Table
## ============================================================================
## SECTION 5.1: Initialize Table Structure
# Create table starting with PET metadata and add empty columns for MRI data
pet_mri_tab = pet_meta[['Subject.ID','Project','Phase' ,'Sex','Weight','Research.Group','VISCODE','Study.Date','Age','Modality','Description','Imaging.Protocol','Image.Data.ID','modality_subtype','PATH']]
pet_mri_tab = pet_mri_tab.copy()
pet_mri_tab['MRI_subID']= ''
pet_mri_tab['MRI_Modality']= ''
pet_mri_tab['MRI_Research.Group']= ''
pet_mri_tab['MRI_Age']= np.nan
pet_mri_tab['MRI_VISCODE']= ''
pet_mri_tab['MRI_studydate']= ''
pet_mri_tab['MRI_Research.Group']= ''
pet_mri_tab['MRI_Description']= ''
pet_mri_tab['MRI_Imaging.protocol']= ''
pet_mri_tab['MRI_Type']= ''
pet_mri_tab['MRI_ID']= ''
pet_mri_tab['MRI_PATH']= ''

# %%
## ============================================================================
## SECTION 5.2: Match PET Scans with MRI Scans
## ============================================================================
# Main pairing algorithm: For each PET scan, find best matching MRI scan
# Matching criteria (in order of preference):
#   1. Same VISCODE
#   2. Same Age
#   3. Age difference <= 1 year
# For ADNI 3 phase, prefer Original over Processed MRI scans
suc = 0 
er = 0

#for i in range(1992):
for i in range(pet_meta.shape[0]):
    pet_id = pet_meta.iloc[i,:]
    
    mri_data = mri_meta.copy()
    
    mri_data= mri_data[mri_data['Subject.ID']==pet_id['Subject.ID']]
                        
    mri_data['age_diff'] = abs(float(pet_id.Age) - mri_data.Age.astype(float))
    
    mri_data= mri_data[((mri_data['Age'] == pet_id['Age'])
                        | (mri_data['VISCODE'] == pet_id['VISCODE'])
                        |(abs(mri_data.age_diff)<=1))]
                          
    if mri_data.shape[0]== 0:
        er += 1
        del mri_data
        print(str(i) + ' '+ pet_id['Subject.ID']+' error')
        continue
        
    ideal_mri = mri_data[mri_data['VISCODE'] == pet_id['VISCODE']]
    if ideal_mri.shape[0] == 0:
        ideal_mri = mri_data[mri_data['Age'] == pet_id['Age']]
    if ideal_mri.shape[0] == 0:
        ideal_mri = mri_data[mri_data.age_diff == mri_data.age_diff.min()]
    if ideal_mri.shape[0] == 0:
        er += 1
        del mri_data
        print(str(i) + ' '+ pet_id['Subject.ID']+' error')
        continue
            
    mri_data_orig = ideal_mri[ideal_mri['Type'] == 'Original']
    mri_data_proc = ideal_mri[ideal_mri['Type'] == 'Processed']
    
    if pet_id['Phase'] == 'ADNI 3' and mri_data_orig.shape[0]>0:
        suc += 1
        pet_mri_tab['MRI_subID'].values[i]= mri_data_orig.iloc[0,:]['Subject.ID']
        pet_mri_tab['MRI_Modality'].values[i]= mri_data_orig.iloc[0,:]['Modality']
        pet_mri_tab['MRI_Research.Group'].values[i]= mri_data_orig.iloc[0,:]['Research.Group']
        pet_mri_tab['MRI_Age'].values[i]= mri_data_orig.iloc[0,:]['Age']
        pet_mri_tab['MRI_VISCODE'].values[i]= mri_data_orig.iloc[0,:]['VISCODE']
        pet_mri_tab['MRI_studydate'].values[i]= mri_data_orig.iloc[0,:]['Study.Date']
        pet_mri_tab['MRI_Description'].values[i]= mri_data_orig.iloc[0,:]['Description']
        pet_mri_tab['MRI_Imaging.protocol'].values[i]= mri_data_orig.iloc[0,:]['Imaging.Protocol']
        pet_mri_tab['MRI_Type'].values[i]= mri_data_orig.iloc[0,:]['Type']
        pet_mri_tab['MRI_ID'].values[i]= mri_data_orig.iloc[0,:]['Image.Data.ID']
        pet_mri_tab['MRI_PATH'].values[i]= mri_data_orig.iloc[0,:]['PATH']

    elif ((pet_id['Phase'] != 'ADNI 3' and mri_data_proc.shape[0]>0) or 
          (pet_id['Project'] == 'A4' and mri_data_proc.shape[0]>0)):
        suc += 1
        pet_mri_tab['MRI_subID'].values[i]= mri_data_proc.iloc[0,:]['Subject.ID']
        pet_mri_tab['MRI_Modality'].values[i]= mri_data_proc.iloc[0,:]['Modality']
        pet_mri_tab['MRI_Research.Group'].values[i]= mri_data_proc.iloc[0,:]['Research.Group']
        pet_mri_tab['MRI_Age'].values[i]= mri_data_proc.iloc[0,:]['Age']
        pet_mri_tab['MRI_VISCODE'].values[i]= mri_data_proc.iloc[0,:]['VISCODE']
        pet_mri_tab['MRI_studydate'].values[i]= mri_data_proc.iloc[0,:]['Study.Date']
        pet_mri_tab['MRI_Description'].values[i]= mri_data_proc.iloc[0,:]['Description']
        pet_mri_tab['MRI_Imaging.protocol'].values[i]= mri_data_proc.iloc[0,:]['Imaging.Protocol']
        pet_mri_tab['MRI_Type'].values[i]= mri_data_proc.iloc[0,:]['Type']
        pet_mri_tab['MRI_ID'].values[i]= mri_data_proc.iloc[0,:]['Image.Data.ID']
        pet_mri_tab['MRI_PATH'].values[i]= mri_data_proc.iloc[0,:]['PATH']


# %%
pet_mri_tab = pet_mri_tab[pet_mri_tab['MRI_Age'].notna()]

# %%
pet_mri_tab = pet_mri_tab[pet_mri_tab['Age']>0]
pet_mri_tab.reset_index(drop=True, inplace = True)

# %%
pet_mri_tab = pet_mri_tab[pet_mri_tab['MRI_Type'] == 'Processed']
pet_mri_tab.reset_index(drop=True, inplace = True)

# %%
if (pet_mri_tab['Subject.ID'] == pet_mri_tab['MRI_subID']).all():
    print("✅ All Subject.ID values match MRI_subID.")
else:
    print("❌ There are mismatches between Subject.ID and MRI_subID.")

# %%
# add path to pet scans normalised to whole cerebellum, script cerebellum_normalisation. 
for i in range(0,np.shape(pet_mri_tab)[0]):
    name_pet = pet_mri_tab['Image.Data.ID'][i]+'_normalised.nii'
    filename = fnmatch.filter(os.listdir('/csc/epitkane/data/ADNI_A4/ADNI_16_04_22_A4_25_10_23_registered_normalised_pet/'), name_pet)
    if len(filename)> 0:
        pet_mri_tab.loc[i,'PET_PATH_normalised'] = '/csc/epitkane/data/ADNI_A4/ADNI_16_04_22_A4_25_10_23_registered_normalised_pet/' + filename[0]
    
    name_mri = pet_mri_tab['MRI_ID'][i]+'_registered.nii'
    filename = fnmatch.filter(os.listdir('/csc/epitkane/data/ADNI_A4/ADNI_16_04_22_A4_25_10_23_registered_mri/'), name_mri)
    if len(filename)> 0:
        pet_mri_tab.loc[i,'MRI_PATH_registered'] = '/csc/epitkane/data/ADNI_A4/ADNI_16_04_22_A4_25_10_23_registered_mri/' + filename[0]
     
    #print(i)

# %%
pet_mri_tab = pet_mri_tab[pet_mri_tab['PET_PATH_normalised'].notna()]
pet_mri_tab.reset_index(drop=True, inplace = True)
pet_mri_tab = pet_mri_tab[pet_mri_tab['MRI_PATH_registered'].notna()]
pet_mri_tab.reset_index(drop=True, inplace = True)

# %%


# %%
# remove scans with bad registration
meta_qc = pd.read_csv('/csc/epitkane/home/atagmazi/ADDL_pipeline/scripts/registration/metafile_ADDLpipeline_abeta_mri_27_05_2025_afterQC.csv',header=[0], index_col=[0])
pet_mri_tab = pet_mri_tab[pet_mri_tab['Image.Data.ID'].isin(meta_qc['Image.Data.ID'])].copy()
pet_mri_tab.reset_index(drop=True, inplace = True)

# %%
#split data on train and test subsets
train_size = 0.8
train_end = int(len(pet_mri_tab)*train_size)
t = int(0.8*train_end) #!!
v = int(0.2*train_end) 
#DATADIR = r"/csc/epitkane/data/ADNI/AD_DL_03_11_2021/"
print(f'train set size = {t}')

# %%

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
def process_row(i, pet_path, mri_path):
    result = {
        'i': i,
        'pet_min': 0,
        'pet_max': 0,
        'pet_90q': 0,
        'pet_95q': 0,
        'pet_99q': 0,
        'pet_999q': 0,
        'mri_min': 0,
        'mri_max': 0,
        'mri_90q': 0,
        'mri_95q': 0,
        'mri_99q': 0,
        'mri_999q': 0,
        'pet_values': [],
        'mri_values': [],
        'invalid': False
    }
    logger.info(f"Processing index {i}")  # Log each index being processed

    try:
        image_pet = np.asarray(nib.load(pet_path).get_fdata())
        image_mri = np.asarray(nib.load(mri_path).get_fdata())
        
        if np.isnan(image_pet).any() or np.isnan(image_mri).any():
            result['invalid'] = True
            return result
        elif np.max(image_pet) <= 0 or np.max(image_mri) <= 0:
            result['invalid'] = True
            return result
        
        result['pet_min'] = np.min(image_pet)
        result['pet_max'] = np.max(image_pet)
        result['pet_90q'] = np.percentile(image_pet, 90)
        result['pet_95q'] = np.percentile(image_pet, 95)
        result['pet_99q'] = np.percentile(image_pet, 99)
        result['pet_999q'] = np.percentile(image_pet, 99.9)
        if i <= t:
            result['pet_values'] = np.random.choice(image_pet.flatten(), size=min(10000, image_pet.size), replace=False)
        
        result['mri_min'] = np.min(image_mri)
        result['mri_max'] = np.max(image_mri)
        result['mri_90q'] = np.percentile(image_mri, 90)
        result['mri_95q'] = np.percentile(image_mri, 95)
        result['mri_99q'] = np.percentile(image_mri, 99)
        result['mri_999q'] = np.percentile(image_mri, 99.9)
        if i <= t:
            result['mri_values'] = np.random.choice(image_mri.flatten(), size=min(10000, image_mri.size), replace=False)
        
    except Exception as e:
        print(f"Error processing index {i}: {e}")
        result['invalid'] = True
    
    return result

# Number of workers to run in parallel (adjust based on your CPU)
n_jobs = 8

results = Parallel(n_jobs=n_jobs)(
    delayed(process_row)(i, pet_mri_tab['PET_PATH_normalised'][i], pet_mri_tab['MRI_PATH_registered'][i])
    for i in range(pet_mri_tab.shape[0])
)

# Rebuild dataframe from results
for res in results:
    i = res['i']
    if not res['invalid']:
        pet_mri_tab.loc[i, 'pet_min'] = res['pet_min']
        pet_mri_tab.loc[i, 'pet_max'] = res['pet_max']
        pet_mri_tab.loc[i, 'pet_90q'] = res['pet_90q']
        pet_mri_tab.loc[i, 'pet_95q'] = res['pet_95q']
        pet_mri_tab.loc[i, 'pet_99q'] = res['pet_99q']
        pet_mri_tab.loc[i, 'pet_999q'] = res['pet_999q']
        
        pet_mri_tab.loc[i, 'mri_min'] = res['mri_min']
        pet_mri_tab.loc[i, 'mri_max'] = res['mri_max']
        pet_mri_tab.loc[i, 'mri_90q'] = res['mri_90q']
        pet_mri_tab.loc[i, 'mri_95q'] = res['mri_95q']
        pet_mri_tab.loc[i, 'mri_99q'] = res['mri_99q']
        pet_mri_tab.loc[i, 'mri_999q'] = res['mri_999q']
        
pet_all_values = np.concatenate([r['pet_values'] for r in results if not r['invalid']])
mri_all_values = np.concatenate([r['mri_values'] for r in results if not r['invalid']])
i_invalid = [r['i'] for r in results if r['invalid']]

# Final min/max stats
print(f"Invalid indexes: {i_invalid}")
print(f"PET min: {pet_mri_tab['pet_min'].min()}, max: {pet_mri_tab['pet_max'].max()}")
print(f"MRI min: {pet_mri_tab['mri_min'].min()}, max: {pet_mri_tab['mri_max'].max()}")


# %%
# Drop invalid rows
pet_mri_tab.drop(index=i_invalid, inplace=True)
pet_mri_tab.reset_index(drop=True, inplace=True)

print(f"Remaining valid rows: {pet_mri_tab.shape[0]}")

# %%
pet_all_values = np.array(pet_all_values)
mri_all_values = np.array(mri_all_values)

p_quant90 = np.quantile(pet_all_values, 0.90)
m_quant90 = np.quantile(mri_all_values, 0.90)

p_quant95 = np.quantile(pet_all_values, 0.95)
m_quant95 = np.quantile(mri_all_values, 0.95)


p_quant99 = np.quantile(pet_all_values, 0.99)
m_quant99 = np.quantile(mri_all_values, 0.99)

p_quant999 = np.quantile(pet_all_values, 0.999)
m_quant999 = np.quantile(mri_all_values, 0.999)

p_mean = np.mean(pet_all_values)
m_mean = np.mean(mri_all_values)

p_std = np.std(pet_all_values)
m_std = np.std(mri_all_values)

p_median = np.median(pet_all_values)
m_median = np.median(mri_all_values)

p_iqr = iqr(pet_all_values, rng=(25, 75))
m_iqr = iqr(mri_all_values, rng=(25, 75))

p_after_clip = np.clip(pet_all_values, 0, p_quant999, out=None)
m_after_clip = np.clip(mri_all_values, 0, m_quant999, out=None)
p_mean_clip = np.mean(p_after_clip)
m_mean_clip = np.mean(m_after_clip)
p_std_clip = np.std(p_after_clip)
m_std_clip = np.std(m_after_clip)
p_min_clip = np.min(p_after_clip)
m_min_clip = np.min(m_after_clip)
p_max_clip = np.max(p_after_clip)
m_max_clip = np.max(m_after_clip)


np.savez("stats_train.npz",
         p_quant90=p_quant90,
         m_quant90=m_quant90,
         p_quant95=p_quant95,
         m_quant95=m_quant95,
         p_quant99=p_quant99, 
         m_quant99=m_quant99,
         p_quant999=p_quant999, 
         m_quant999=m_quant999,
         p_mean=p_mean, 
         m_mean=m_mean, 
         p_std=p_std, 
         m_std=m_std,
         p_median=p_median, 
         m_median=m_median, 
         p_iqr=p_iqr, 
         m_iqr=m_iqr,
         p_mean_clip=p_mean_clip, 
         m_mean_clip=m_mean_clip,
         p_std_clip=p_std_clip, 
         m_std_clip=m_std_clip,
         p_mim_clip=p_min_clip, 
         m_min_clip=m_min_clip,
         p_max_clip=p_max_clip, 
         m_max_clip=m_max_clip
        )


# %%
plt.figure(figsize=(10, 5))

# PET Histogram
plt.subplot(2, 2, 1)
plt.hist(pet_all_values, bins=500, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.title("PET Image Histogram")

# MRI Histogram
plt.subplot(2, 2, 2)
plt.hist(mri_all_values, bins=500, color='red', alpha=0.7, edgecolor='black')
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.title("MRI Image Histogram")


plt.subplot(2, 2, 3)
plt.hist(np.log1p(pet_all_values), bins=500, color='blue', alpha=0.7, edgecolor='black')  # Log scale on Y-axis
plt.xlabel("Log Pixel Value")
plt.ylabel("Frequency")
plt.title("PET Image Histogram")



plt.subplot(2, 2, 4)
plt.hist(np.log1p(mri_all_values), bins=500, color='red', alpha=0.7, edgecolor='black')
plt.xlabel("Log Pixel Value")
plt.ylabel("Frequency")
plt.title("MRI Image Histogram")


# Save and show the plot
plt.tight_layout()
plt.savefig("pixels_value_histograms.png")  # Save the figure
plt.show()  # Display the plot

# %%
plt.figure(figsize=(10, 5))

# PET Histogram
plt.subplot(2, 2, 1)
plt.hist(p_after_clip, bins=500, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.title("PET Image Histogram")

# MRI Histogram
plt.subplot(2, 2, 2)
plt.hist(m_after_clip, bins=500, color='red', alpha=0.7, edgecolor='black')
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.title("MRI Image Histogram")


plt.subplot(2, 2, 3)
plt.hist(np.log1p(p_after_clip), bins=500, color='blue', alpha=0.7, edgecolor='black')  # Log scale on Y-axis
plt.xlabel("Log Pixel Value")
plt.ylabel("Frequency")
plt.title("PET Image Histogram")



plt.subplot(2, 2, 4)
plt.hist(np.log1p(m_after_clip), bins=500, color='red', alpha=0.7, edgecolor='black')
plt.xlabel("Log Pixel Value")
plt.ylabel("Frequency")
plt.title("MRI Image Histogram")


# Save and show the plot
plt.tight_layout()
plt.savefig("pixels_value_histograms_clip999.png")  # Save the figure
plt.show()  # Display the plot

# %%


# %%


# %%
#pet_mri_tab = pet_mri_tab[pet_mri_tab.mri_min >= 0]
pet_mri_tab.reset_index(drop=True, inplace = True)

# %%


# %%
pet_minimum = np.min(pet_mri_tab['pet_min'])
pet_maximum = np.max(pet_mri_tab['pet_max'])
mri_minimum = np.min(pet_mri_tab['mri_min'])
mri_maximum = np.max(pet_mri_tab['mri_max'])
print(f'MinMax computation is done! PET minimum is {pet_minimum}, PET maximum is {pet_maximum}. MRI minimum is {mri_minimum}, MRI maximum is {mri_maximum}')


# %%


# %%
pet_mri_tab.to_csv('pet_mri_pairs.csv')

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
pet_mri_tab['Research.Group'].value_counts().values

# %%
pet_mri_tab['Research.Group'].value_counts()

# %%
plt.figure(figsize=(15,6))

ax = sns.violinplot(x="Research.Group", y="Age", data=pet_mri_tab)

# Calculate number of obs per group & median to position labels
medians = pet_mri_tab.groupby(['Research.Group'])['Age'].max().values
nobs = pet_mri_tab['Research.Group'].value_counts(sort=False).values
nobs = [str(x) for x in nobs.tolist()]
nobs = ["n: " + i for i in nobs]
 
# Add text to the figure
pos = range(len(nobs))
for tick, label in zip(pos, ax.get_xticklabels()):
   ax.text(pos[tick], medians[tick] + 0.03, nobs[tick],
            horizontalalignment='left',
            size=13,
            color='blue',
            weight='semibold')
#plt.show()
plt.savefig('samples_dist.png')

# %%
pd.set_option('display.max_rows', None)
pet_mri_tab[pet_mri_tab.Project == 'A4'].Age.value_counts()

# %%


# %%



