# EC500-Project-Kidney-Localization
This repository is for BU ENG EC500 Course ( Medical Imaging with AI) Project - Kidney Localization with Limited Data

## Team Members
Zachary Loschinskey, Omri Yosfan, Maolin Wei, Shashank Basty

## Introduction
Our task is to detect and draw a bounding box around the kidney(s) in medical images generated through diffusion weighted MRI to pave the way for improved DW-MRI imaging enhancing diagnosis, treatment, and monitoring of renal diseases, including:
- Develop an algorithm to accurately detect the right and left kidneys.
  - We will develop our methods based on the approch - Segment Anything in Medical Images ([MedSAM](https://github.com/bowang-lab/MedSAM))
  - We will modify the network architecture, replace loss function used for segmentation task with the loss functions used for object detection task, and re-train the model based on our dataset.
- Locate a bounding box, large enough (no too tight), allowing for motion, particularly in superior-inferior direction.
- Develop evaluation code to evaluate the model performance with standard object detection metrics such as Intersection over Union (IoU) and Mean Average Precision (mAP).
- Develop another detection algorithm using U-Net for comparison.
  - See U-Nets Folder for more information in README 

## Installation
1. Create a virtual environment `conda create -n [env_name] python=3.10` and activate it `conda activate [env_name]`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`

## Dataset
The dataset includes 8 diffusion weighted MRI acquisitions collected in healthy volunteers. Each acquisition is coronal, allowing for easy viewing of the kidneys. Additionally, there is a held-out dataset that will be used to evaluate our model at the end of the project.

Each volunteer (case1 to case8) contains 5 volumn images (Fimage_AP_0163.nrrd, ... , Fimage_AP_0167.nrrd). Every volumn image share the same masks of kidneys (svr_leftKidneyMask2.nii.gz and svr_rightKidneyMask2.nii.gz)

**Visualization of original sample data of our dataset:**

<img src="https://github.com/Maolin-Wei/EC500-Project-Kidney-Localization/assets/144057115/b735a742-6ce6-4c9f-ad67-3bdf13104399" width="270" height="270" alt="Data Sample">

It is too dark and not in the correct direction, so it will be processed in the later processes.

## MedSAM + Detection Model
### Data Pre-processing
**1. Format dataset**  
  This process is to ensure each image file has a corresponding label file.
  
  Since the original data share the same masks for each image in the same case. It needs to be processed to make each image correspond to a single mask, and the seperate masks for left kidney and right kidney will be integrated into one single mask.
   
  For example, for case1 image (Fimage_AP_0163.nrrd) and mask (svr_leftKidneyMask2.nii.gz and svr_rightKidneyMask2.nii.gz). After the processing, image Fimage_AP_0163_case1.nrrd and its corresponding label Fimage_AP_0163_case1.nii.gz will be created.
```bash
python create_dataset.py
```

**2. Create data for training/validation/testing**  
  To meet the input of MedSAM, each 3D image data and corresponding label will be processed to 2D slices with the shape of `1024 x 1024 x 3`. The 2D slice images and labels will be saved as `npy` file.
```bash
python data_preprocess.py
```
- Clip the intensity values to the range between the `0.5th` and `99.5th` percentiles for MRI images.
- Utilize max-min normalization to rescale the intensity values to `[0, 255]`.
- Resample image and label size to `1024 x 1024 x 3`

**Visualization of sample data after pre-processing:**

![data_smaple_after_preprocessing](https://github.com/Maolin-Wei/EC500-Project-Kidney-Localization/assets/144057115/42fca3f1-954e-45cf-be0d-2725f46825ee)

**The data will be rotated 90 degrees counterclockwise when loading during training and evaluation**

![data_smaple_after_preprocessing](https://github.com/Maolin-Wei/EC500-Project-Kidney-Localization/assets/144057115/62e8ffa2-8107-447f-b6a3-dbc7012f1fb4)

### Training
Download [MedSAM model checkpoint](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN) and place it at e.g., `MedSAM/checkpoint/MedSAM/medsam_vit_b.pth`

```bash
python train.py --tr_npy_path /path/to/training_data --val_npy_path /path/to/validation_data --checkpoint /path/to/MedSAM_checkpoint.pth
```

More parameters can be seen in the `train.py`

### Evaluation
To evaluate the model, specify the `model_path`, `data_path` in the `evaluate.py`, then run:

```bash
python evaluate.py
```

## Acknowledgements
- We highly appreciate the [MedSAM](https://github.com/bowang-lab/MedSAM) team for the great foundation model.
- We also thank Sila Kurugol for proposing this project.
