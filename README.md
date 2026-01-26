# petVAE
Amyloid positron emission tomography (PET) scans are used to define amyloid-β (Aβ) accumulation, the core biomarker of Alzheimer’s disease (AD).  Typically, individuals are classified as either Aβ-negative or Aβ-positive. However, we developed the model, petVAE, which latent features effectively represent the AD continuum and defined biologically meaningful clusters. Based on petVAE latent features we were able to definde subgroups of Aβ-negative or Aβ-positive individuals that are differ by genetic risk allell, AD biomarkers level or cognitive performence.

# About the model (in process)
![plot](petVAE_architecture.png)
The petVAE is a 2D convolutional variational autoencoder that contains 1.10 million parameters. 

Input:
The model was trained on [18F]-AV45 (Florbetapir) Amyloid PET scans pre-registered to correspondant MRI and T1 MNI152 template with dimensions 182×218×182 voxels. For best performances it's recommended to use the same registration template and dimensions of the input PET scan. However, due to padding stage of the model it can operate on scans with axial size smaller than 256x256 (PET scan size < 256x256x[anything])

Activation function: ReLU

Optimization: Adam optimization algorithm (Kingma & Ba, 2014) with an initial learning rate of 0.0001

Evaluation metrics: mean squared error (MSE)

Number of epochs: maximum of 150 epochs and stopped early, if loss in the validation dataset did not decrease for 15 epochs 

Batch size: 4

For increasing the training set size we used augmentation approach. We applied either gaussian noise (σ=0,5,10,15,20,25%) or flipped images by X or Y axes with equal probability.



# How to use the model
1. Clone ArcheD repository from Github.
2. **In the folder 'model_to_use'** unzip the file model.zip.
3. Run `pip install arched_package.zip`
4. Now you can run ArcheD model with your command line.

```  
arched [-h] [--output_name OUTPUT_NAME] path_to_directory folder_with_scans

 A novel residual neural network for predicting amyloid CSF directly from amyloid PET scans

 positional arguments:
  path_to_directory     the path to folder that contains model (model_08-0.12_20_10_22.h5), arched_package.zip and folder with PET scans, for ex. '~/(your path)/model_to_use/'
  folder_with_scans     the name of the folder with scans (if the folder with scans is in path_to_directory) or the full path to it, for ex. 'scans' (as it locates in model_to_use folder) or '~/(your path)/scans'

 optional arguments:
  -h, --help            show this help message and exit
  --output_name OUTPUT_NAME, -o OUTPUT_NAME
                        name for the output file, for ex. 'arched_amyloid_csf_prediction'. Note: include the path if you want the output file to be saved not in the path_to_directory.
```

Example of the command line

`arched '~/model_to_use/' 'scans' -o 'arched_amyloid_csf_prediction'` 

5. If the model runs successfully, you will get the 'Model run successfully!' message and the CSV file will appear in your working directory. The file name will consist of the 'output_name', time and date of the model running.

# Authors
Arina A. Tagmazian, Claudia Schwarz, Catharina Lange, Esa Pitkänen, Eero Vuoksimaa

Data used for training and evaluation the model were obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) and Anti-Amyloid Treatment in Asymptomatic Alzheimer's Disease (A4/LEARN) studies. 

Preprint of the manuscript with results is available on [BioRxiv](https://will be here later). 
