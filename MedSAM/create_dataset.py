import os
import shutil
import nibabel as nib
import numpy as np

data_folder = '/projectnb/ec500kb/projects/Project_3/Maolin/Data/'
output_folder = './data/KidneyData/'

'''
Create dataset by each case.
Since each case has 5 images but share the same mask, 
we need to create the mask file seperately for each image and make it have the corresponding file name, 
and we will save the image to images_folder, and combine the two masks into one and save to the labels_folder

eg: 
For case1 image (Fimage_AP_0163.nrrd) and mask (svr_leftKidneyMask2.nii.gz and svr_rightKidneyMask2.nii.gz),
after the processing, images_folder will has image file Fimage_AP_0163_case1.nrrd, etc.
and labels_folder will have corresponding label file Fimage_AP_0163_case1.nii.gz, etc.
'''

images_folder = os.path.join(output_folder, 'images')
labels_folder = os.path.join(output_folder, 'labels')

for case_folder in os.listdir(data_folder):
    case_path = os.path.join(data_folder, case_folder)
    if os.path.isdir(case_path):
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(labels_folder, exist_ok=True)

        # Iterate through each image file in the case folder
        for image_file in os.listdir(case_path):
            if image_file.startswith('Fimage_AP_') and image_file.endswith('.nrrd'):
                image_path = os.path.join(case_path, image_file)

                # Load the left and right kidney masks using SimpleITK
                left_kidney_mask_path = os.path.join(case_path, 'svr_leftKidneyMask2.nii.gz')
                right_kidney_mask_path = os.path.join(case_path, 'svr_rightKidneyMask2.nii.gz')

                left_kidney_mask = nib.load(left_kidney_mask_path).get_fdata()
                right_kidney_mask = nib.load(right_kidney_mask_path).get_fdata()

                integrated_mask = np.zeros_like(left_kidney_mask)
                integrated_mask[left_kidney_mask > 0] = 1  # left kidney has label 1
                integrated_mask[right_kidney_mask > 0] = 2  # right kedney has label 2
                integrated_mask_img = nib.Nifti1Image(integrated_mask, affine=None)

                combined_mask_name = image_file.replace('.nrrd', f'_{case_folder}.nii.gz')
                combined_mask_path = os.path.join(labels_folder, combined_mask_name)
                nib.save(integrated_mask_img, combined_mask_path)

                # Copy the image file to the output images folder
                new_image_name = f"{image_file.split('.')[0]}_{case_folder}.nrrd"
                shutil.copy(image_path, os.path.join(images_folder, new_image_name))