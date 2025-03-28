import os
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import greycomatrix, greycoprops
import networkx as nx
import random

# =============================================================================
# ROI Extraction Function
# =============================================================================
def extract_non_cancer_rois(image, mask, roi_width, roi_height, overlap, max_rois=None):
    """
    Extract non-cancer ROIs from the image with specified overlap.
    """
    h, w = image.shape
    stride_y = int(roi_height * (1 - overlap))
    stride_x = int(roi_width * (1 - overlap))
    
    rois = []
    for y in range(0, h - roi_height + 1, stride_y):
        for x in range(0, w - roi_width + 1, stride_x):
            roi = image[y:y+roi_height, x:x+roi_width]
            roi_mask = mask[y:y+roi_height, x:x+roi_width]
            # Only add ROI if it is completely non-cancer (mask sum == 0)
            if np.sum(roi_mask) == 0:
                rois.append(roi)
            if max_rois and len(rois) >= max_rois:
                return rois
    return rois

# =============================================================================
# Dataset Classes
# =============================================================================
class BladderCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for main_folder in os.listdir(root_dir):
            main_path = os.path.join(root_dir, main_folder)
            if not os.path.isdir(main_path):
                continue
            if main_folder == 'Redo':
                for sub_folder in os.listdir(main_path):
                    sub_path = os.path.join(main_path, sub_folder)
                    if os.path.isdir(sub_path):
                        self._process_folder(sub_path, f"Redo-{sub_folder}")
            else:
                self._process_folder(main_path, main_folder)

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
                for file in os.listdir(case_path):
                    if file.endswith('.dcm'):
                        dcm_file = os.path.join(case_path, file)
                    elif file.endswith('.png'):
                        mask_file = os.path.join(case_path, file)
                if dcm_file and mask_file:
                    self.samples.append({
                        'dcm': dcm_file,
                        'mask': mask_file,
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
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return {
            'image': image,
            'mask': mask,
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
                    'image': roi,
                    'time_point': sample['time_point'],
                    'ct_folder': sample['ct_folder'],
                    'case_type': sample['case_type']
                })
        return roi_samples

    def __len__(self):
        return len(self.roi_samples)

    def __getitem__(self, idx):
        sample = self.roi_samples[idx]
        roi_tensor = torch.from_numpy(sample['image']).float().unsqueeze(0)
        return {
            'image': roi_tensor,
            'time_point': sample['time_point'],
            'ct_folder': sample['ct_folder'],
            'case_type': sample['case_type']
        }

# =============================================================================
# Visualization Class for ROI
# =============================================================================
class BladderCancerROIVisualizer:
    @staticmethod
    def visualize_single_roi(roi_sample):
        roi = roi_sample['image'].squeeze().numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(roi, cmap='gray')
        plt.title(f"ROI\n{roi_sample['time_point']} - {roi_sample['ct_folder']} - {roi_sample['case_type']}")
        plt.axis('off')
        plt.show()

    @staticmethod
    def visualize_roi_batch(batch, num_samples=4):
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
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            roi = sample['image'].squeeze().numpy()
            axes[i].imshow(roi, cmap='gray')
            axes[i].set_title(f"{sample['time_point']}\n{sample['ct_folder']} - {sample['case_type']}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

# =============================================================================
# GLCM Feature Extraction using scikit-image
# =============================================================================
class GLCMFeatureExtractor:
    @staticmethod
    def compute_glcm(image, levels=256, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """
        Compute the GLCM using scikit-image.
        """
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max()-image.min()) * (levels-1)).astype(np.uint8)
        glcm = greycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
        return glcm

    @staticmethod
    def extract_glcm_contrast(glcm):
        """
        Extract the contrast property from the GLCM.
        """
        contrast = greycoprops(glcm, 'contrast')
        return np.mean(contrast)

    def extract_contrast_feature(self, image):
        glcm = self.compute_glcm(image)
        return self.extract_glcm_contrast(glcm)

# =============================================================================
# Build Rule-Based Network using GLCM Contrast Feature
# =============================================================================
def build_roi_network(roi_dataset, glcm_threshold=10):
    """
    Build a network where each ROI is a node with a GLCM contrast feature.
    Nodes are connected if they share the same time_point and case_type and have a contrast difference
    below the specified threshold.
    """
    glcm_extractor = GLCMFeatureExtractor()
    G = nx.DiGraph()
    for idx in range(len(roi_dataset)):
        sample = roi_dataset[idx]
        roi_image = sample['image'].squeeze().numpy()
        contrast_feature = glcm_extractor.extract_contrast_feature(roi_image)
        G.add_node(idx, glcm_contrast=contrast_feature,
                   time_point=sample['time_point'],
                   ct_folder=sample['ct_folder'],
                   case_type=sample['case_type'])
    for i in range(len(roi_dataset)):
        for j in range(len(roi_dataset)):
            if i != j:
                if (G.nodes[i]['time_point'] == G.nodes[j]['time_point'] and
                    G.nodes[i]['case_type'] == G.nodes[j]['case_type']):
                    if abs(G.nodes[i]['glcm_contrast'] - G.nodes[j]['glcm_contrast']) < glcm_threshold:
                        G.add_edge(i, j)
    return G

# =============================================================================
# Gradient-Based Random Walk on the Network
# =============================================================================
def network_gradient_walk(G, max_walk_length=10, max_walks=50, mode=1, attribute_name="glcm_contrast"):
    """
    Perform a gradient-based random walk on network G.
    """
    attr = nx.get_node_attributes(G, attribute_name)
    nodes = list(G.nodes)
    random_walk_graphs = []
    walks_done = 0
    while walks_done < max_walks:
        current_node = random.choice(nodes)
        path = [current_node]
        path_desc = []
        for _ in range(max_walk_length):
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                path_desc.append('no_neighbors')
                break
            grad_scores = {}
            for neighbor in neighbors:
                grad_scores[neighbor] = (attr[neighbor] - attr[current_node]) / (attr[current_node] + 1e-8)
            max_val = max(grad_scores.values())
            best_neighbors = [n for n, score in grad_scores.items() if score == max_val]
            path_desc.append('_')
            current_node = random.choice(best_neighbors)
            path.append(current_node)
        G_rw = nx.DiGraph()
        nx.add_path(G_rw, path)
        desc_dict = dict(zip(path, path_desc))
        nx.set_node_attributes(G_rw, desc_dict, "path_description")
        random_walk_graphs.append(G_rw)
        walks_done += 1
    return random_walk_graphs

# =============================================================================
# Main Pipeline: Data Loading, ROI Extraction, Network Building, and Random Walk
# =============================================================================
def main_pipeline():
    # Load base dataset and create ROI dataset
    base_dataset = BladderCancerDataset(root_dir='data/preprocessed/Al-Bladder Cancer/')
    roi_dataset = BladderCancerROIDataset(
        base_dataset, 
        roi_width=128, 
        roi_height=128, 
        overlap=0.40, 
        max_rois_per_image=10
    )
    
    # Visualize one ROI sample (optional)
    visualizer = BladderCancerROIVisualizer()
    sample_roi = roi_dataset[0]
    visualizer.visualize_single_roi(sample_roi)
    
    # Build the ROI network using GLCM contrast
    G = build_roi_network(roi_dataset, glcm_threshold=10)
    print("Network has {} nodes and {} edges".format(G.number_of_nodes(), G.number_of_edges()))
    
    # Visualize the network structure
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("ROI Network Based on GLCM Contrast")
    plt.show()
    
    # Perform gradient-based random walks on the network
    rw_graphs = network_gradient_walk(G, max_walk_length=10, max_walks=50, mode=1, attribute_name="glcm_contrast")
    for i, rw in enumerate(rw_graphs[:3]):
        plt.figure(figsize=(8, 4))
        pos_rw = nx.spring_layout(rw)
        nx.draw(rw, pos_rw, with_labels=True, node_color='salmon', edge_color='black')
        plt.title("Random Walk {}".format(i+1))
        plt.show()

if __name__ == '__main__':
    main_pipeline()
