import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def extract_non_cancer_rois(image, mask, roi_width, roi_height, overlap, max_rois=None):
    """
    Extract non-cancer ROIs from the image with specified overlap.
    
    Args:
    image (numpy.ndarray): Input image array.
    mask (numpy.ndarray): Binary mask where 1 indicates cancer regions.
    roi_width (int): Width of the ROI to extract.
    roi_height (int): Height of the ROI to extract.
    overlap (float): Overlap between ROIs, value between 0 and 1.
    max_rois (int, optional): Maximum number of ROIs to extract. If None, extract all possible ROIs.
    
    Returns:
    list: List of extracted ROI arrays.
    """
    h, w = image.shape
    stride_y = int(roi_height * (1 - overlap))
    stride_x = int(roi_width * (1 - overlap))
    
    rois = []
    for y in range(0, h - roi_height + 1, stride_y):
        for x in range(0, w - roi_width + 1, stride_x):
            roi = image[y:y+roi_height, x:x+roi_width]
            roi_mask = mask[y:y+roi_height, x:x+roi_width]
            
            # Check if the ROI is completely non-cancer
            if np.sum(roi_mask) == 0:
                rois.append(roi)
            
            if max_rois and len(rois) >= max_rois:
                return rois
    
    return rois

class BladderCancerROIVisualizer:
    @staticmethod
    def visualize_single_roi(roi_sample):
        """
        Visualize a single ROI sample.
        
        Args:
        roi_sample (dict): A dictionary containing ROI data and metadata.
        """
        roi = roi_sample['image'].squeeze().numpy()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(roi, cmap='gray')
        plt.title(f"ROI\n{roi_sample['time_point']} - {roi_sample['ct_folder']} - {roi_sample['case_type']}")
        plt.axis('off')
        plt.show()

    @staticmethod
    def visualize_roi_batch(batch, num_samples=4):
        """
        Visualize a batch of ROI samples.
        
        Args:
        batch (dict): A batch of ROI samples.
        num_samples (int): Number of samples to visualize from the batch.
        """
        batch_size = batch['image'].shape[0]
        num_samples = min(num_samples, batch_size)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
        
        for i in range(num_samples):
            roi = batch['image'][i].squeeze().numpy()
            
            axes[i].imshow(roi, cmap='gray')
            axes[i].set_title(f"{batch['time_point'][i]}\n{batch['ct_folder'][i]}\n{batch['case_type'][i]}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_dataset(dataset, num_samples=4):
        """
        Visualize random samples from the dataset.
        
        Args:
        dataset (BladderCancerROIDataset): The ROI dataset.
        num_samples (int): Number of random samples to visualize.
        """
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
        
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            roi = sample['image'].squeeze().numpy()
            
            axes[i].imshow(roi, cmap='gray')
            axes[i].set_title(f"{sample['time_point']}\n{sample['ct_folder']}\n{sample['case_type']}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
