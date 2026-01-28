# %%
# run_model.py
import argparse
import torch
import nibabel as nib



# %%
class PETSliceDataset(Dataset):
    def __init__(self, list_IDs_pet, slice_axis=2, brain_mask=None, 
                 pet_minimum= p_min_clip, pet_maximum=p_max_clip,
                 mri_minimum= m_min_clip, mri_maximum=m_max_clip,
                 pet_quant = p_quant999,mri_quant = m_quant999, 
                 pet_mean = p_mean_clip,mri_mean = m_mean_clip,
                 pet_std = p_std_clip,mri_std = m_std_clip,
                 sagittal_dim=182, coronal_dim=218, axial_dim=182):
        """
        PyTorch Dataset for paired 2D slices of PET and MRI scans.
        """
        self.list_IDs_pet = list_IDs_pet
        self.slice_axis = slice_axis  # 0 = sagittal, 1 = coronal, 2 = axial
        self.brain_mask = brain_mask
        
        self.pet_minimum = pet_minimum
        self.pet_maximum = pet_maximum
        self.pet_quant = pet_quant
        self.pet_mean = pet_mean
        self.pet_std = pet_std
        self.mri_minimum = mri_minimum
        self.mri_maximum = mri_maximum
        self.mri_quant = mri_quant
        self.mri_mean = mri_mean
        self.mri_std = mri_std
        
        self.sagittal_dim = sagittal_dim
        self.coronal_dim = coronal_dim
        self.axial_dim = axial_dim
        self.slices = self.load_all_slices()  # Preload slice paths
        self.indices = list(range(len(self.slices)))

    def load_all_slices(self):
        """Extracts and pairs 2D slices from all PET/MRI scans."""
        slices = []
        slice_id = 0
        for pet_path in zip(self.list_IDs_pet):
            if self.slice_axis == 0:  # Sagittal
                num_slices = slice_id + self.sagittal_dim 
            elif self.slice_axis == 1:  # Coronal
                num_slices = slice_id + self.coronal_dim 
            else:  # Axial (default)
                num_slices = slice_id + self.axial_dim 

            for within_img_num, i in enumerate(range(slice_id, num_slices)):
                slices.append((pet_path, i, within_img_num))  # Store slice index
            slice_id = num_slices
        return slices

    def __len__(self):
        """Returns the number of slices."""
        return len(self.slices)

    def __data_generation(self, batch_slices):
        """Generates one batch of 2D slices."""
        pet_slices = []
        pet_ids = []
        batch_data = []

        #pet_path, slice_idx, slice_num_inimg = batch_slices[0]
        for slice_info in batch_slices:
            pet_path, slice_idx, slice_num_inimg = slice_info  # Ensure correct unpacking
            pet_path = pet_path[0]
            if not isinstance(pet_path, str):  
                print(pet_path)
                print(slice_idx)
                print(slice_num_inimg)
                raise ValueError(f"Expected pet_path to be a string, got {type(pet_path)}")
                
        
        img_pet = nib.load(pet_path).get_fdata()

            # Extract the corresponding 2D slice
        if self.slice_axis == 0:  # Sagittal
            pet_slice = img_pet[slice_num_inimg, :, :]
            if self.brain_mask is not None:
                bm = self.brain_mask[slice_num_inimg, :, :]
                pet_slice *= bm
                    
        elif self.slice_axis == 1:  # Coronal
            pet_slice = img_pet[:, slice_num_inimg, :]
            if self.brain_mask is not None:
                bm = self.brain_mask[:, slice_num_inimg, :]
                pet_slice *= bm
            
        else:  # Axial (default)
            pet_slice = img_pet[:, :, slice_num_inimg]
            if self.brain_mask is not None:
                bm = self.brain_mask[:, :, slice_num_inimg]
                pet_slice *= bm
                
            

            # Normalize if necessary (optional step, currently not applied)
        
        pet_norm = self.min_max_normalize(np.asarray(pet_slice, dtype=np.float32), float(self.pet_quant), self.pet_minimum, self.pet_maximum)
        
        pet_norm = np.clip(pet_norm, 1e-7, 1 - 1e-7) #???or just to 0 and 1?
        
        # Convert to NumPy arrays
        
        batch_data = np.array(pet_norm, dtype=np.float32)
        #print(batch_data.shape[0])
        batch_data = batch_data.reshape(1, batch_data.shape[0],  batch_data.shape[1])
        #print(batch_data.shape)
        #if len(batch_data) == 0:
            #return None, None, None
        

        # Convert to tensor
        return batch_data, pet_path, slice_num_inimg 

    def __getitem__(self, index):
        """Generates one slice of 2D images (per slice, not batch)."""
        # Get a single slice metadata
        pet_path, slice_idx, slice_num_inimg = self.slices[index]
        
        # Generate single slice data
        X, pet_ids, slice_n = self.__data_generation([(pet_path, slice_idx, slice_num_inimg)])
        X = torch.tensor(X, dtype=torch.float32)
        #print(X.shape)


        return { 'image': X, 'pet_ID': pet_ids, 'slice_number': slice_n }
    
    def min_max_normalize(self,image,quantile, min_val, max_val):
        image = np.asarray(image, dtype=np.float32).copy()  # Ensure array and prevent in-place modification
        image = np.clip(image, 0, quantile)  # Clip values above quantile
        return (image - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero


# %%
#standart loss function
def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean') # mean of mse losses within batch 
    logvar = torch.clamp(logvar, max=5, min = -5)
    KLD = 0 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1) # or torch.mean()?
    return recon_loss,torch.mean(KLD)
   


# %%
def save_scan_outputs(scan_id, recon_slices, lat_features, df, out_dir):
    """
    Reconstruct 3D volume in correct slice order
    and concatenate latent features in the same order.
    """

    # ---- Sort by slice number ----
    recon_slices = sorted(recon_slices, key=lambda x: x[0])
    lat_features = sorted(lat_features, key=lambda x: x[0])

    # ---- Stack reconstructed slices ----
    volume = np.stack([s[1] for s in recon_slices], axis=-1)
    affine = np.eye(4)  # replace with original affine if available
    nib.save(
        nib.Nifti1Image(volume, affine),
        f"{out_dir}/{scan_id}_recon.nii.gz"
    )

    # ---- Concatenate latent vectors ----
    latent_vector = np.concatenate([s[1] for s in lat_features])

    # one column per scan
    df[scan_id] = latent_vector

    print(f"Saved scan and latent features for {scan_id}")


# %%
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------- Load brain mask ----------
    brain_mask = nib.load(args.brainmask).get_fdata()
    brain_mask[brain_mask != 0] = 1

    dataset = PETSliceDataset(
        list_IDs_pet=args.input,
        brain_mask=brain_mask
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.numworkers
    )

    # ---------- Load trained model ----------
    petVAE=torch.load("petVAE_model.pth")
    petVAE.to(device)
    petVAE.eval()

    df = pd.DataFrame()

    current_id = None
    recon_slices = []      # store (slice_num, recon_slice)
    lat_features = []      # store (slice_num, mu_vector)

    with torch.no_grad():
        for data in dataloader:

            # Extract proper scalar values
            ID = data['pet_ID'][0]
            slice_num = int(data['slice_number'][0])
            img = data['image'].float().to(device)

            # ---------- Run model ----------
            recon_batch, mu, logvar = petVAE(img)

            mu_np = mu.squeeze().cpu().numpy()
            recon_np = recon_batch.squeeze().cpu().numpy()

            # ---------- Initialize first scan ----------
            if current_id is None:
                current_id = ID

            # ---------- If new scan starts → save previous ----------
            if ID != current_id:
                save_scan_outputs(
                    current_id,
                    recon_slices,
                    lat_features,
                    df,
                    args.output_dir
                )

                # reset buffers
                recon_slices = []
                lat_features = []
                current_id = ID

            # ---------- Accumulate current slice ----------
            recon_slices.append((slice_num, recon_np))
            lat_features.append((slice_num, mu_np))

        # ---------- Save LAST scan ----------
        save_scan_outputs(
            current_id,
            recon_slices,
            lat_features,
            df,
            args.output_dir
        )

    # ---------- Save latent features ----------
    df.to_csv(args.latfeatures_out)
    print("Latent features saved!")
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--brainmask", required=True)
    parser.add_argument("--latfeatures_out", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--numworkers", type=int, default=1)
    args = parser.parse_args()
    main(args)


