import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from skimage.measure import label, regionprops

class ROIExtractor:
    def __init__(self, dicom_path: str, mask_path: str):
        self.dcm = pydicom.dcmread(dicom_path)
        self.image = self.dcm.pixel_array
        self.mask = np.array(Image.open(mask_path))
        
        if np.max(self.mask) > 1:
            self.mask = (self.mask > 0).astype(np.uint8)
    
    def get_cancer_roi_bbox(self) -> List[Tuple[int, int, int, int]]:
        """Get bounding boxes for cancer ROIs"""
        labeled_mask = label(self.mask)
        regions = regionprops(labeled_mask)
        
        bboxes = []
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            bboxes.append((minr, minc, maxr, maxc))
        
        return bboxes
    
    def extract_non_cancer_rois(self, roi_size: Tuple[int, int], 
                               method: str = 'non_overlap',
                               overlap_ratio: float = 0.5) -> List[np.ndarray]:
        """
        Extract non-cancer regions of specified size
        
        Args:
            roi_size: Tuple of (height, width) for ROI
            method: 'non_overlap', 'sliding_window', or 'random'
            overlap_ratio: For sliding window, how much overlap between windows
        """
        height, width = roi_size
        non_cancer_mask = 1 - self.mask
        
        rois = []
        
        if method == 'non_overlap':
            for i in range(0, self.image.shape[0] - height, height):
                for j in range(0, self.image.shape[1] - width, width):
                    roi = self.image[i:i+height, j:j+width]
                    mask_roi = non_cancer_mask[i:i+height, j:j+width]
                    
                    if np.all(mask_roi):  # If all pixels are non-cancer
                        rois.append(roi)
        
        elif method == 'sliding_window':
            stride_h = max(1, int(height * (1 - overlap_ratio)))
            stride_w = max(1, int(width * (1 - overlap_ratio)))
            
            for i in range(0, self.image.shape[0] - height, stride_h):
                for j in range(0, self.image.shape[1] - width, stride_w):
                    roi = self.image[i:i+height, j:j+width]
                    mask_roi = non_cancer_mask[i:i+height, j:j+width]
                    
                    if np.all(mask_roi):
                        rois.append(roi)
        
        elif method == 'random':
            max_attempts = 1000
            num_desired = len(self.get_cancer_roi_bbox()) * 2  # 2x as many non-cancer as cancer ROIs
            
            for _ in range(max_attempts):
                if len(rois) >= num_desired:
                    break
                    
                i = np.random.randint(0, self.image.shape[0] - height)
                j = np.random.randint(0, self.image.shape[1] - width)
                
                roi = self.image[i:i+height, j:j+width]
                mask_roi = non_cancer_mask[i:i+height, j:j+width]
                
                if np.all(mask_roi):
                    rois.append(roi)
        
        return rois
    
    def extract_all_rois(self, method: str = 'non_overlap', 
                         overlap_ratio: float = 0.5) -> Dict[str, List[np.ndarray]]:
        """Extract both cancer and non-cancer ROIs"""
        cancer_rois = []
        cancer_bboxes = self.get_cancer_roi_bbox()
        
        # Extract cancer ROIs
        for bbox in cancer_bboxes:
            minr, minc, maxr, maxc = bbox
            roi = self.image[minr:maxr, minc:maxc]
            cancer_rois.append(roi)
        
        # Get the average size of cancer ROIs
        avg_height = int(np.mean([roi.shape[0] for roi in cancer_rois]))
        avg_width = int(np.mean([roi.shape[1] for roi in cancer_rois]))
        
        # Extract non-cancer ROIs of similar size
        non_cancer_rois = self.extract_non_cancer_rois(
            (avg_height, avg_width), method, overlap_ratio
        )
        
        return {
            'cancer': cancer_rois,
            'non_cancer': non_cancer_rois
        }
    
    def visualize_rois(self, rois: Dict[str, List[np.ndarray]], num_samples: int = 5):
        """Visualize extracted ROIs"""
        num_cancer = min(num_samples, len(rois['cancer']))
        num_non_cancer = min(num_samples, len(rois['non_cancer']))
        
        fig, axes = plt.subplots(2, max(num_cancer, num_non_cancer), 
                                figsize=(15, 6))
        
        for i in range(num_cancer):
            axes[0, i].imshow(rois['cancer'][i], cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Cancer ROI {i+1}')
        
        for i in range(num_non_cancer):
            axes[1, i].imshow(rois['non_cancer'][i], cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f'Non-Cancer ROI {i+1}')
        
        plt.tight_layout()
        plt.show()