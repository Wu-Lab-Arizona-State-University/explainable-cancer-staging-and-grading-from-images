import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ROIExtractor import extract_non_cancer_rois,BladderCancerROIVisualizer
import pydicom
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

class BladderCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for main_folder in os.listdir(root_dir):
            main_path = os.path.join(root_dir, main_folder)
            print("Main Path: ",main_path)
            if not os.path.isdir(main_path):
                continue

            if main_folder == 'Redo':
                for sub_folder in os.listdir(main_path):
                    sub_path = os.path.join(main_path, sub_folder)
                    if os.path.isdir(sub_path):
                        self._process_folder(sub_path, f"Redo-{sub_folder}")
            else:
                self._process_folder(main_path, main_folder)
                print("TO,T1+:")

    def _process_folder(self, folder_path, time_point):
        for ct_folder in os.listdir(folder_path):
            ct_path = os.path.join(folder_path, ct_folder)
            if not os.path.isdir(ct_path):
                continue

            for case_type in ['Control', 'Lesion']:
                case_path = os.path.join(ct_path, case_type)
                if not os.path.isdir(case_path):
                    continue

                dcm_file = None
                mask_file = None
                # coords_file = None

                for file in os.listdir(case_path):
                    if file.endswith('.dcm'):
                        dcm_file = os.path.join(case_path, file)
                        print("Redo: ",dcm_file)
                    elif file.endswith('.png'):
                        mask_file = os.path.join(case_path, file)
                    # elif file == 'coords.txt':
                    #     coords_file = os.path.join(case_path, file)

                if dcm_file and mask_file: #and coords_file:
                    self.samples.append({
                        'dcm': dcm_file,
                        'mask': mask_file,
                        # 'coords': coords_file,
                        'time_point': time_point,
                        'ct_folder': ct_folder,
                        'case_type': case_type
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        dcm = pydicom.dcmread(sample['dcm'])
        image = dcm.pixel_array.astype(np.float32)
        
        image = (image - image.min()) / (image.max() - image.min())

        mask = np.array(Image.open(sample['mask'])).astype(np.float32)
        mask = mask / 255.0 

        # with open(sample['coords'], 'r') as f:
        #     coords = f.read().strip()

        image = torch.from_numpy(image).unsqueeze(0)  
        mask = torch.from_numpy(mask).unsqueeze(0)

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return {
            'image': image,
            'mask': mask,
            # 'coords': coords,
            'time_point': sample['time_point'],
            'ct_folder': sample['ct_folder'],
            'case_type': sample['case_type']
        }

class BladderCancerROIDataset(Dataset):
    def __init__(self, base_dataset, roi_width, roi_height, overlap, max_rois_per_image=None):
        self.base_dataset = base_dataset
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.overlap = overlap
        self.max_rois_per_image = max_rois_per_image
        
        self.roi_samples = self._extract_all_rois()

    def _extract_all_rois(self):
        roi_samples = []
        for idx in range(len(self.base_dataset)):
            sample = self.base_dataset[idx]
            image = sample['image'].squeeze().numpy()
            mask = sample['mask'].squeeze().numpy()
            
            rois = extract_non_cancer_rois(image, mask, self.roi_width, self.roi_height, 
                                           self.overlap, self.max_rois_per_image)
            
            for roi in rois:
                roi_samples.append({
                    'roi': roi,
                    'time_point': sample['time_point'],
                    'ct_folder': sample['ct_folder'],
                    'case_type': sample['case_type']
                })
        
        return roi_samples

    def __len__(self):
        return len(self.roi_samples)

    def __getitem__(self, idx):
        sample = self.roi_samples[idx]
        
        # Convert ROI to tensor and add channel dimension
        roi_tensor = torch.from_numpy(sample['roi']).float().unsqueeze(0)
        
        return {
            'image': roi_tensor,
            'time_point': sample['time_point'],
            'ct_folder': sample['ct_folder'],
            'case_type': sample['case_type']
        }

class BladderCancerVisualizer:
    @staticmethod
    def visualize_sample(sample):
        image = sample['image'].squeeze().numpy()
        mask = sample['mask'].squeeze().numpy()
        
        bbox = BladderCancerVisualizer.compute_bounding_box(mask)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title(f"DICOM Image\n{sample['time_point']} - {sample['ct_folder']} - {sample['case_type']}")
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')
        
        axes[2].imshow(image, cmap='gray')
        axes[2].add_patch(plt.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0],
                                        fill=False, edgecolor='red', linewidth=1))
        axes[2].set_title('Image with Bounding Box')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_batch(batch, num_samples=4):
        batch_size = batch['image'].shape[0]
        num_samples = min(num_samples, batch_size)
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        
        for i in range(num_samples):
            image = batch['image'][i].squeeze().numpy()
            mask = batch['mask'][i].squeeze().numpy()
            bbox = BladderCancerVisualizer.compute_bounding_box(mask)
            
            axes[i, 0].imshow(image, cmap='gray')
            axes[i, 0].set_title(f"DICOM Image\n{batch['time_point'][i]} - {batch['ct_folder'][i]} - {batch['case_type'][i]}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Mask')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(image, cmap='gray')
            axes[i, 2].add_patch(plt.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0],
                                               fill=False, edgecolor='red', linewidth=1))
            axes[i, 2].set_title('Image with Bounding Box')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def compute_bounding_box(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, cmin, rmax, cmax

def visualize_dataset(dataset, num_samples=4):
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    for i in range(num_samples):
        sample = dataset[i]
        image = sample['image'].squeeze().numpy()
        mask = sample['mask'].squeeze().numpy()
        bbox = BladderCancerVisualizer.compute_bounding_box(mask)
        
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title(f"DICOM Image\n{sample['time_point']} - {sample['ct_folder']} - {sample['case_type']}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Mask')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(image, cmap='gray')
        axes[i, 2].add_patch(plt.Rectangle((bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0],
                                           fill=False, edgecolor='red', linewidth=2))
        axes[i, 2].set_title('Image with Bounding Box')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
        
# dataset = BladderCancerDataset(root_dir ='/home/as-aravinthakshan/Desktop/RESEARCH/explainable-cancer-staging-and-frading-from-images/data/preprocessed/Al-Bladder Cancer')
# # sample = dataset[200]  # Get the first sample
# # BladderCancerVisualizer.visualize_sample(sample)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
# batch = next(iter(dataloader))
# BladderCancerVisualizer.visualize_batch(batch)


# base_dataset = BladderCancerDataset(root_dir='data/preprocessed/Al-Bladder Cancer/')
# roi_dataset = BladderCancerROIDataset(
#     base_dataset, 
#     roi_width=128, 
#     roi_height=128, 
#     overlap=0.25, 
#     max_rois_per_image=10
# )

# roi_dataloader = DataLoader(
#     roi_dataset,
#     batch_size=16,
#     shuffle=True,
#     num_workers=4
# )

# visualizer = BladderCancerROIVisualizer()

# sample = roi_dataset[0]
# visualizer.visualize_single_roi(sample)

# batch = next(iter(roi_dataloader))
# visualizer.visualize_roi_batch(batch)

# visualizer.visualize_dataset(roi_dataset)