# U-Net Implementation
This folder contains files used to define and train our task-specific U-Nets to compare to MedSAM

### Files:
**2D_Unet.ipynb:** Extracts slices from dataset, augments training set, trains model, tests model and outputs metrics.

**2D_Unet_Shuffling.ipynb:** Extends upon 2D_Unet to train models with every possible combination of test/train split for given test set size (Number of subjects desired)

**3D_Unet.ipynb:** Extracts volumes from dataset, augments training set, trains model, tests model and outputs metrics.

**3D_Unet_Shuffling:** Extends upon 3D_Unet to train models with every possible combination of test/train split for given test set size (Number of subjects desired)
