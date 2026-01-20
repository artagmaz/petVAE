# %% [markdown]
# # PET VAE Model Architecture Definitions

# %%
# Core PyTorch imports for neural network construction
import torch
import torch.nn as nn
import torch.nn.functional as F
# subprocess is only used for converting notebook to script (Cell 40) - can be removed if not needed
import subprocess

# %%
class ConvBlock(nn.Module):
    """
    Convolutional block for the encoder: downsamples input by factor of 2.
    
    This block performs a 2D convolution with stride=2 to reduce spatial dimensions,
    followed by LeakyReLU activation. Batch normalization is commented out (not used).
    
    Parameters:
    - nb_filters_in (int): Number of input channels
    - nb_filters_out (int): Number of output channels
    """
    def __init__(self, nb_filters_in, nb_filters_out):
        super(ConvBlock, self).__init__()
        # Padding calculation: (kernel_size // 2) ensures output size is exactly half when stride=2
        self.padding = (3 // 2, 3 // 2)  # Results in padding=(1, 1)
        self.conv = nn.Conv2d(in_channels=nb_filters_in, out_channels=nb_filters_out, 
                             kernel_size=(3, 3), padding=self.padding, stride=2)
        # Batch normalization was tested but not used in final model
        #self.normalize = nn.BatchNorm2d(nb_filters_out)
        self.activation = nn.LeakyReLU()
        # Alternative activation (not used):
        #self.activation = nn.ReLU()

    def forward(self, x):
        xhat = self.conv(x)
        # Batch normalization step (commented out - not used)
        #xhat = self.normalize(xhat)
        xhat = self.activation(xhat)
        return xhat


class DeconvBlock(nn.Module):
    """
    Deconvolutional (transposed convolution) block for the decoder: upsamples input by factor of 2.
    
    This block performs a 2D transposed convolution with stride=2 to increase spatial dimensions,
    followed by LeakyReLU activation. Used in the decoder to reconstruct images from latent space.
    
    Parameters:
    - nb_filters_in (int): Number of input channels
    - nb_filters_out (int): Number of output channels
    """
    def __init__(self,  nb_filters_in, nb_filters_out):
        super(DeconvBlock, self).__init__()
        # Padding calculation: (kernel_size // 2) ensures output size is exactly double when stride=2
        self.padding = (3 // 2, 3 // 2)  # Results in padding=(1, 1)
        self.deconv = nn.ConvTranspose2d(in_channels=nb_filters_in, out_channels=nb_filters_out, 
                                        kernel_size=(3, 3), padding=self.padding, stride=2)
        # Batch normalization was tested but not used in final model
        #self.normalize = nn.BatchNorm2d(nb_filters_out)
        self.activation = nn.LeakyReLU()
        # Alternative activation (not used):
        #self.activation = nn.ReLU()

    def forward(self, x):
        xhat = self.deconv(x)
        # Batch normalization step (commented out - not used)
        #xhat = self.normalize(xhat)
        xhat = self.activation(xhat)
        return xhat



# %%
class ImagePadding_4dtensor:
    def __init__(self, image):
        """
        Initializes the padding class with a 4D tensor (batch_size, channels, height, width).
        """
        # Ensure that the input is a 4D tensor (batch_size, channels, height, width)
        assert len(image.shape) == 4, "Input image must have shape (batch_size, channels, height, width)"
        self.image = image

    def pad_to_size(self, target_size):
        """
        Pads the input 2D image batch to the target size by adding pixels with zero values.

        Parameters:
        - target_size (tuple): The desired output size (height, width)

        Returns:
        - Padded 4D tensor (batch_size, channels, target_height, target_width)
        """
        current_height, current_width = self.image.shape[2], self.image.shape[3]

        # Calculate padding needed for each dimension
        pad_height = max(target_size[0] - current_height, 0)
        pad_width = max(target_size[1] - current_width, 0)

        # Divide padding into two sides (before and after)
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Apply padding using torch.nn.functional.pad
        padded_image = F.pad(
            self.image,
            (pad_left, pad_right, pad_top, pad_bottom),  # (left, right, top, bottom)
            mode='constant',
            value=0  # Pad with zeros
        )

        return padded_image


# %%
class ImagePadding_3dtensor:
    def __init__(self, image):
        """
        Initializes the padding class with a 3D tensor (batch_size, height, width).
        """
        # Ensure that the input is a 3D tensor (batch_size, height, width)
        assert len(image.shape) == 3, "Input image must have shape (batch_size, height, width)"
        self.image = image

    def pad_to_size(self, target_size):
        """
        Pads the input 2D image batch to the target size by adding pixels with zero values.

        Parameters:
        - target_size (tuple): The desired output size (height, width)

        Returns:
        - Padded 3D tensor (batch_size, target_height, target_width)
        """
        current_height, current_width = self.image.shape[1], self.image.shape[2]

        # Calculate padding needed for each dimension
        pad_height = max(target_size[0] - current_height, 0)
        pad_width = max(target_size[1] - current_width, 0)

        # Divide padding into two sides (before and after)
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Apply padding using torch.nn.functional.pad
        padded_image = F.pad(
            self.image.unsqueeze(1),  # Add channel dimension (B, 1, H, W) for padding
            (pad_left, pad_right, pad_top, pad_bottom),  # (left, right, top, bottom)
            mode='constant',
            value=0  # Pad with zeros
        ).squeeze(1)  # Remove extra channel dimension to get back (B, H, W)

        return padded_image




# %%
class ImageCropping_4dtensor:
    def __init__(self, image):
        """
        Initializes the cropping class with a 4D tensor (batch_size, channels, height, width).
        Assumes input shape is (batch_size, channels, height, width).
        """
        # Ensure that the input has a batch dimension and has 2 channels
        assert len(image.shape) == 4, "Input image must have shape (batch_size, channels, height, width)"
        self.image = image

    def crop_to_size(self, target_size):
        """
        Crops the input 2D image batch to the target size.

        Parameters:
        - target_size (tuple): The desired output size (height, width)

        Returns:
        - Cropped 4D tensor (batch_size, channels, target_height, target_width)
        """
        current_height, current_width = self.image.shape[2], self.image.shape[3]

        # Ensure the target size is smaller or equal to the current size
        assert target_size[0] <= current_height and target_size[1] <= current_width, \
            "Target size must be smaller or equal to the current size in both dimensions."

        # Calculate cropping indices
        crop_top = (current_height - target_size[0]) // 2
        crop_left = (current_width - target_size[1]) // 2

        # Crop the image while keeping the batch and channel structure
        cropped_image = self.image[:, :, crop_top:crop_top + target_size[0], crop_left:crop_left + target_size[1]]

        return cropped_image


# %%
class ImageCropping_3dtensor:
    def __init__(self, image):
        """
        Initializes the cropping class with a 3D tensor (batch_size, height, width).
        Assumes input shape is (batch_size, height, width).
        """
        # Ensure the input has a batch dimension
        assert len(image.shape) == 3, "Input image must have shape (batch_size, height, width)"
        self.image = image

    def crop_to_size(self, target_size):
        """
        Crops the input 2D image batch to the target size.

        Parameters:
        - target_size (tuple): The desired output size (height, width)

        Returns:
        - Cropped 3D tensor (batch_size, target_height, target_width)
        """
        current_height, current_width = self.image.shape[1], self.image.shape[2]

        # Ensure the target size is smaller or equal to the current size
        assert target_size[0] <= current_height and target_size[1] <= current_width, \
            "Target size must be smaller or equal to the current size in both dimensions."

        # Calculate cropping indices
        crop_top = (current_height - target_size[0]) // 2
        crop_left = (current_width - target_size[1]) // 2

        # Crop the image while keeping the batch structure
        cropped_image = self.image[:, crop_top:crop_top + target_size[0], crop_left:crop_left + target_size[1]]

        return cropped_image




# %%
# MAIN MODEL CLASS - Single-modality VAE for PET images
# This is the primary model used in training (petVAE_model_training.ipynb)
# 
class VAE_1modality_PET(nn.Module):
    """
    Variational Autoencoder for single-modality PET image reconstruction.
    
    Architecture:
    - Encoder: 2 ConvBlocks (1->32, 32->64 channels) + Flatten + 2 Linear layers (mean, log_var)
    - Latent space: feature_size -> latent_size
    - Decoder: Linear layer + 3 DeconvBlocks + final ConvTranspose2d + ReLU
    
    Input: PET images of shape (batch, 1, 182, 218) - padded to (256, 256) internally
    Output: Reconstructed PET images of shape (batch, 1, 182, 218)
    
    Parameters:
    - feature_size (int): Size of flattened feature map after encoder (64*64*64 = 262144)
    - latent_size (int): Dimension of latent space (typically 64)
    - in_channels (int): Number of input channels (default: 1 for PET)
    """
    def __init__(self, feature_size, latent_size, in_channels=1):
        super(VAE_1modality_PET, self).__init__()
        
        self.feature_size = feature_size  # Size after flattening encoder output
        self.latent_size = latent_size    # Latent space dimension
        self.in_channels = in_channels    # Input channels (1 for PET)

        # ENCODER: Convolutional layers to extract features
        # Each ConvBlock reduces spatial dimensions by 2x (stride=2)
        # Input: (B, 1, 256, 256) -> (B, 32, 128, 128) -> (B, 64, 64, 64)
        self.conv = nn.Sequential(
            ConvBlock(1, 32),   # First downsampling: 256x256 -> 128x128
            ConvBlock(32, 64)   # Second downsampling: 128x128 -> 64x64
        )
        
        # Flatten spatial dimensions before passing to fully connected layers
        self.flatten = nn.Flatten()
        
        # Latent space projection: feature_size -> latent_size
        # These output the mean and log-variance of the latent distribution
        self.dense_mean = nn.Linear(self.feature_size, self.latent_size)  
        self.dense_log_var = nn.Linear(self.feature_size, self.latent_size)  
        
        # DECODER: Reconstruct from latent space
        # First, project latent vector back to feature space
        self.dense = nn.Linear(self.latent_size, self.feature_size)
        
        # Then, deconvolve back to image space
        # Each DeconvBlock increases spatial dimensions by 2x (stride=2)
        # (B, 64, 64, 64) -> (B, 32, 127, 127) -> (B, 32, 253, 253) -> (B, 32, 505, 505)
        # Final layer: (B, 32, 505, 505) -> (B, 1, 505, 505), then cropped to (182, 218)
        self.deconv = nn.Sequential(
            DeconvBlock(64, 32),   # First upsampling
            DeconvBlock(32, 32),   # Second upsampling
            DeconvBlock(32, 32),   # Third upsampling
            # Final layer: no upsampling, just channel reduction with stride=1
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()  # Final activation (ensures non-negative output)
        )




    def encode(self, x):
        """
        Encodes input PET image into latent space parameters (mean and log-variance).
        
        Process:
        1. Pad input to (256, 256) for consistent processing
        2. Apply convolutional encoder (downsampling)
        3. Flatten spatial dimensions
        4. Project to latent space (mean and log-variance)
        
        Parameters:
        - x (Tensor): Input PET image of shape (batch, 1, 182, 218)
        
        Returns:
        - z_mu (Tensor): Mean of latent distribution, shape (batch, latent_size)
        - z_var (Tensor): Log-variance of latent distribution, shape (batch, latent_size)
        """
        # Pad input to target size (256, 256) for consistent encoder processing
        # Input images are typically (182, 218) and need padding
        image_padder = ImagePadding_4dtensor(x)
        target_size = (256, 256)
        padded_image = image_padder.pad_to_size(target_size)
        
        # Apply convolutional encoder: (B, 1, 256, 256) -> (B, 64, 64, 64)
        x = self.conv(padded_image)
        
        # Store shape for decoder (needed to reshape flattened features back)
        self.remember_shape = x.shape  # (batch, 64, 64, 64)
        
        # Flatten spatial dimensions: (B, 64, 64, 64) -> (B, 262144)
        x = self.flatten(x)
        
        # Project to latent space: (B, feature_size) -> (B, latent_size)
        z_mu = self.dense_mean(x)      # Mean of latent distribution
        z_var = self.dense_log_var(x)  # Log-variance of latent distribution
        
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: samples from latent distribution.
        
        Instead of sampling directly from N(mu, var), we sample epsilon from N(0,1)
        and compute z = mu + epsilon * std. This makes the sampling differentiable.
        
        Parameters:
        - mu (Tensor): Mean of latent distribution
        - logvar (Tensor): Log-variance of latent distribution
        
        Returns:
        - z (Tensor): Sampled latent vector, shape (batch, latent_size)
        """
        std = torch.exp(0.5 * logvar)  # Convert log-variance to standard deviation
        eps = torch.randn_like(std)     # Sample epsilon from N(0,1)
        return mu + eps * std           # Reparameterization: z = mu + epsilon * std

    def decode(self, z):
        """
        Decodes latent vector back to PET image.
        
        Process:
        1. Project latent vector to feature space
        2. Reshape to spatial dimensions (using remembered shape from encoder)
        3. Apply deconvolutional decoder (upsampling)
        4. Crop back to original input size (182, 218)
        
        Parameters:
        - z (Tensor): Latent vector of shape (batch, latent_size)
        
        Returns:
        - cropped_image (Tensor): Reconstructed PET image of shape (batch, 1, 182, 218)
        """
        # Project latent vector back to feature space: (B, latent_size) -> (B, feature_size)
        x = self.dense(z)
        
        # Reshape to spatial dimensions: (B, feature_size) -> (B, 64, 64, 64)
        # Uses shape remembered from encoder forward pass
        x = torch.reshape(x, (self.remember_shape[0], self.remember_shape[1], 
                             self.remember_shape[2], self.remember_shape[3]))
        
        # Apply deconvolutional decoder: (B, 64, 64, 64) -> (B, 1, 505, 505)
        x = self.deconv(x)
        
        # Crop back to original input size: (B, 1, 505, 505) -> (B, 1, 182, 218)
        image_cropper = ImageCropping_4dtensor(x)
        target_size = (182, 218)
        cropped_image = image_cropper.crop_to_size(target_size)
        
        return cropped_image

    def forward(self, x):
        """
        Forward pass: encode input, sample from latent space, decode to reconstruction.
        
        Parameters:
        - x (Tensor): Input PET image of shape (batch, 1, 182, 218)
        
        Returns:
        - recon_x (Tensor): Reconstructed image, shape (batch, 1, 182, 218)
        - mu (Tensor): Mean of latent distribution, shape (batch, latent_size)
        - logvar (Tensor): Log-variance of latent distribution, shape (batch, latent_size)
        """
        # Encode to latent space
        mu, logvar = self.encode(x)
        # Sample from latent distribution using reparameterization trick
        z = self.reparameterize(mu, logvar)
        # Decode to reconstruction
        return self.decode(z), mu, logvar


# %%
# UTILITY CELL - Converts notebook to Python script
# This cell converts the notebook to a .py script for import in other notebooks
# Only needed when architecture changes are made
# Can be commented out if not actively developing
subprocess.run(['jupyter', 'nbconvert', '--to', 'script', 'addl_models_bimodel_pytorch.ipynb'], check=True)

# %%



