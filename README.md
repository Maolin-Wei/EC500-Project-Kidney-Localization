# EC500-Project-Kidney-Localization
This repository is for BU ENG EC500 Course ( Medical Imaging with AI) Project - Kidney Localization with Limited Data

## Introduction
Our task is to detect and draw a bounding box around the kidney(s) in medical images generated through diffusion weighted MRI to pave the way for improved DW-MRI imaging enhancing diagnosis, treatment, and monitoring of renal diseases, including:
- Develop an algorithm to accurately detect the right and left kidneys.
  - We will develop our methods based on the approch - Segment Anything in Medical Images ([MedSAM](https://github.com/bowang-lab/MedSAM))
  - We will modify the network architecture and loss function used for segmentation task to for object detection task and re-train the model based on our dataset.
- Locate a bounding box, large enough (no too tight), allowing for motion, particularly in superior-inferior direction.
- Develop evaluation code to evaluate the model performance with standard object detection metrics such as Intersection over Union (IoU) and Mean Average Precision (mAP).
- Develop another detection algorithm using U-Net as the base model for comparison.

## Installation
1. Create a virtual environment `conda create -n [env_name] python=3.10` and activate it `conda activate [env_name]`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`

## Get Started
### Dataset
The dataset includes 8 diffusion weighted MRI acquisitions collected in healthy volunteers. Each acquisition is coronal, allowing for easy viewing of the kidneys. Additionally, there is a held-out dataset that will be used to evaluate our model at the end of the project.

![sample_data](data/sample_image.png)

### Model Training and Evaluation
This section will be updated soon!

## Acknowledgements
- We highly appreciate the [MedSAM](https://github.com/bowang-lab/MedSAM) team for the great foundation model.
- We also thank Sila Kurugol for proposing this project.
