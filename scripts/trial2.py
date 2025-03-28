import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import networkx as nx
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Dataloader import BladderCancerDataset, BladderCancerROIDataset
import cv2
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import seaborn as sns

class RuleBadedNetwork:
    def __init__(self, num_features, learning_rate=0.01, random_walk_iterations=100):
        """
        Initialize the rule-based network with random walk capabilities
        
        Args:
            num_features (int): Number of input features
            learning_rate (float): Learning rate for weight optimization
            random_walk_iterations (int): Number of iterations for random walk
        """
        self.weights = np.random.randn(num_features)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.random_walk_iterations = random_walk_iterations
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def predict(self, features):
        """
        Predict class based on features
        
        Args:
            features (np.array): Input features
        
        Returns:
            int: Predicted class (0 or 1)
        """
        z = np.dot(features, self.weights) + self.bias
        return 1 if self.sigmoid(z) > 0.5 else 0
    
    def train(self, X, y):
        """
        Train the network using gradient descent with random walk principles
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
        """
        for _ in range(self.random_walk_iterations):
            gradient_weights = np.zeros_like(self.weights)
            gradient_bias = 0
            
            for i in range(len(X)):
                prediction = self.sigmoid(np.dot(X[i], self.weights) + self.bias)
                error = y[i] - prediction
                
                # Add randomness to gradient
                noise_weights = np.random.normal(0, 0.1, self.weights.shape)
                noise_bias = np.random.normal(0, 0.1)
                
                gradient_weights += error * X[i] + noise_weights
                gradient_bias += error + noise_bias
            
            # Update weights and bias
            self.weights += self.learning_rate * gradient_weights / len(X)
            self.bias += self.learning_rate * gradient_bias / len(X)
    
    def random_walk_classification(self, graph, start_node, max_steps=10, exploration_prob=0.2):
        """
        Perform random walk on the network graph for classification
        
        Args:
            graph (nx.Graph): Network graph
            start_node (int): Starting node for random walk
            max_steps (int): Maximum number of steps in random walk
            exploration_prob (float): Probability of exploring a new path
        
        Returns:
            int: Predicted class after random walk
        """
        # Validate start node
        if start_node not in graph.nodes():
            raise ValueError(f"Start node {start_node} not in graph")
        
        current_node = start_node
        visited_nodes = []
        node_features = []
        
        for step in range(max_steps):
            # Collect features from current node
            current_features = graph.nodes[current_node]['features']
            node_features.append(current_features)
            visited_nodes.append(current_node)
            
            # Get neighbors
            neighbors = list(graph.neighbors(current_node))
            
            # Exploration vs exploitation
            if not neighbors or (random.random() < exploration_prob):
                # Random restart or exploration
                break
            
            # Select next node (weighted by similarity)
            next_node = self._select_next_node(graph, current_node, neighbors)
            current_node = next_node
        
        # Aggregate features and predict
        if node_features:
            aggregated_features = np.mean(node_features, axis=0)
            prediction = self.predict(aggregated_features)
            
            return prediction
        
        return None
    
    def _select_next_node(self, graph, current_node, neighbors):
        """
        Select next node in random walk
        
        Args:
            graph (nx.Graph): Network graph
            current_node (int): Current node
            neighbors (list): List of neighbor nodes
        
        Returns:
            int: Selected next node
        """
        # Compute similarity scores (based on node features)
        current_features = graph.nodes[current_node]['features']
        
        # Compute cosine similarity
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Compute similarities and normalize
        similarities = [
            cosine_similarity(current_features, graph.nodes[neighbor]['features']) 
            for neighbor in neighbors
        ]
        
        # Softmax to convert similarities to probabilities
        exp_similarities = np.exp(similarities)
        probabilities = exp_similarities / np.sum(exp_similarities)
        
        # Select node based on probability distribution
        return np.random.choice(neighbors, p=probabilities)


class PretrainedFeatureExtractor:
    def __init__(self, model_name='resnet18', pretrained=True):
        """
        Initialize a pre-trained feature extractor
        
        Args:
            model_name (str): Name of the pre-trained model
            pretrained (bool): Whether to use pre-trained weights
        """
        # Load pre-trained model
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            # Remove the last fully connected layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Set to evaluation mode
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
    def extract_features(self, roi, mask=None):
        """
        Extract features from ROI with optional mask processing
        
        Args:
            roi (torch.Tensor): Region of Interest image
            mask (torch.Tensor, optional): Mask for the ROI
        
        Returns:
            np.ndarray: Extracted features
        """
        # Ensure 3-channel input for pre-trained model
        if roi.dim() == 2 or roi.shape[0] == 1:
            roi = roi.repeat(3, 1, 1)
        
        # Apply mask if provided
        if mask is not None:
            # Normalize mask to [0, 1]
            mask = mask.float()
            roi = roi * mask
        
        # Prepare input for pre-trained model
        input_tensor = roi.unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(input_tensor)
        
        # Convert to numpy and flatten
        features = features.cpu().numpy().flatten()
        
        return features

class MaskFeatureEnhancer:
    @staticmethod
    def extract_mask_features(mask):
        """
        Extract features specifically from the mask
        
        Args:
            mask (torch.Tensor): Binary mask
        
        Returns:
            dict: Mask-specific features
        """
        # Ensure mask is numpy array
        if torch.is_tensor(mask):
            mask = mask.squeeze().numpy()
        
        # Mask-specific features
        mask_features = {
            'area_ratio': np.sum(mask) / mask.size,  # Proportion of mask area
            'perimeter_area_ratio': float(cv2.arcLength(mask.astype(np.uint8), True) / np.sum(mask)),
            'compactness': (np.sum(mask)**2) / (4 * np.pi * np.sum(mask)),
            'boundary_complexity': float(cv2.arcLength(mask.astype(np.uint8), True) / np.sqrt(np.sum(mask)))
        }
        
        return mask_features

class RuleBadedNetwork:
    def __init__(self, num_features, learning_rate=0.01, random_walk_iterations=100):
        """
        Initialize the rule-based network
        
        Args:
            num_features (int): Number of input features
            learning_rate (float): Learning rate for weight optimization
            random_walk_iterations (int): Number of iterations for random walk
        """
        self.weights = np.random.randn(num_features)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.random_walk_iterations = random_walk_iterations
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def predict(self, features):
        """
        Predict class based on features
        
        Args:
            features (np.array): Input features
        
        Returns:
            int: Predicted class (0 or 1)
        """
        z = np.dot(features, self.weights) + self.bias
        return 1 if self.sigmoid(z) > 0.5 else 0
    
    def train(self, X, y):
        """
        Train the network using gradient descent with random walk principles
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
        """
        for _ in range(self.random_walk_iterations):
            gradient_weights = np.zeros_like(self.weights)
            gradient_bias = 0
            
            for i in range(len(X)):
                prediction = self.sigmoid(np.dot(X[i], self.weights) + self.bias)
                error = y[i] - prediction
                
                # Add randomness to gradient
                noise_weights = np.random.normal(0, 0.1, self.weights.shape)
                noise_bias = np.random.normal(0, 0.1)
                
                gradient_weights += error * X[i] + noise_weights
                gradient_bias += error + noise_bias
            
            # Update weights and bias
            self.weights += self.learning_rate * gradient_weights / len(X)
            self.bias += self.learning_rate * gradient_bias / len(X)

def prepare_dataset(roi_dataset, feature_extractor):
    """
    Prepare dataset for network training
    
    Args:
        roi_dataset (BladderCancerROIDataset): ROI dataset
        feature_extractor (PretrainedFeatureExtractor): Feature extraction model
    
    Returns:
        tuple: Prepared features, labels, and graph
    """
    features = []
    labels = []
    masks = []
    
    # Create a graph to represent relationships between ROIs
    G = nx.Graph()
    
    for idx in range(len(roi_dataset)):
        sample = roi_dataset[idx]
        roi = sample['image']
        
        # For test split, we won't use mask information in classification
        # Determine label based on case type
        label = 1 if sample['case_type'] == 'Lesion' else 0
        
        # Extract features using pre-trained model
        feature_vector = feature_extractor.extract_features(roi)
        
        features.append(feature_vector)
        labels.append(label)
        
        # Add node to graph with features
        G.add_node(idx, features=feature_vector, label=label)
    
    # Add edges based on similar time points or CT folders
    for i in range(len(roi_dataset)):
        for j in range(i+1, len(roi_dataset)):
            sample_i = roi_dataset[i]
            sample_j = roi_dataset[j]
            
            # Connect nodes with similar characteristics
            if (sample_i['time_point'] == sample_j['time_point'] or 
                sample_i['ct_folder'] == sample_j['ct_folder']):
                G.add_edge(i, j)
    
    # Scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    return features, np.array(labels), G

def evaluate_model(y_true, y_pred):
    """
    Comprehensive model evaluation
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
    
    Returns:
        dict: Evaluation metrics
    """
    # Compute evaluation metrics
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }
    
    # Visualize confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred))
    
    return metrics

def main(roi_dataset):
    """
    Main function to train and evaluate the rule-based network
    
    Args:
        roi_dataset (BladderCancerROIDataset): ROI dataset
    """
    # Initialize feature extractor
    feature_extractor = PretrainedFeatureExtractor()
    
    # Prepare dataset
    features, labels, graph = prepare_dataset(roi_dataset, feature_extractor)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Initialize and train network
    network = RuleBadedNetwork(num_features=features.shape[1])
    network.train(X_train, y_train)
    
    # Predict on test set
    y_pred = [network.predict(x) for x in X_test]
    
    # Evaluate performance
    metrics = evaluate_model(y_test, y_pred)
    
    # Print metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Perform random walk classification on graph
    print("\nRandom Walk Classification Demonstration:")
    for _ in range(5):  # Try 5 random walks
        start_node = random.choice(list(graph.nodes()))
        walk_prediction = network.random_walk_classification(graph, start_node)
        print(f"Random Walk from Node {start_node}: Predicted Class = {walk_prediction}")

# Metric Explanations
"""
Evaluation Metrics Explained:

1. Accuracy: 
   - Proportion of correct predictions (both true positives and true negatives)
   - Range: 0 to 1 (0% to 100%)
   - Formula: (True Positives + True Negatives) / Total Predictions

2. Precision: 
   - Proportion of true positive predictions among all positive predictions
   - Answers: "Of all instances predicted as positive, how many were actually positive?"
   - Range: 0 to 1
   - Formula: True Positives / (True Positives + False Positives)

3. Recall (Sensitivity): 
   - Proportion of actual positive instances correctly identified
   - Answers: "Of all actual positive instances, how many did we correctly identify?"
   - Range: 0 to 1
   - Formula: True Positives / (True Positives + False Negatives)

4. F1 Score:
   - Harmonic mean of Precision and Recall
   - Provides a single score that balances both metrics
   - Particularly useful for imbalanced datasets
   - Range: 0 to 1
   - Formula: 2 * (Precision * Recall) / (Precision + Recall)

Confusion Matrix Interpretation:
- Top-left: True Negatives (correctly predicted non-lesion)
- Bottom-right: True Positives (correctly predicted lesion)
- Top-right: False Positives (non-lesion predicted as lesion)
- Bottom-left: False Negatives (lesion predicted as non-lesion)
"""

base_dataset = BladderCancerDataset(root_dir='data/preprocessed/Al-Bladder Cancer/')
roi_dataset = BladderCancerROIDataset(
    base_dataset, 
    roi_width=128, 
    roi_height=128, 
    overlap=0.40, 
    max_rois_per_image=10
)
main(roi_dataset)