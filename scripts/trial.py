import os
import numpy as np
import torch
import networkx as nx
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
import random
from Dataloader import BladderCancerDataset, BladderCancerROIDataset
class FeatureExtractor:
    @staticmethod
    def extract_features(roi):
        """
        Extract statistical and textural features from ROI
        
        Args:
            roi (numpy.ndarray): Region of Interest image
        
        Returns:
            dict: Dictionary of extracted features
        """
        # Ensure roi is a numpy array
        if torch.is_tensor(roi):
            roi = roi.squeeze().numpy()
        
        features = {
            # Statistical features
            'mean': np.mean(roi),
            'std': np.std(roi),
            'median': np.median(roi),
            'min': np.min(roi),
            'max': np.max(roi),
            
            # Intensity distribution features
            'skewness': float(np.mean(((roi - np.mean(roi)) / np.std(roi))**3)),
            'kurtosis': float(np.mean(((roi - np.mean(roi)) / np.std(roi))**4)),
            
            # Edge and gradient features
            'gradient_magnitude': np.mean(np.abs(np.gradient(roi))),
            'edge_variance': np.var(np.abs(np.gradient(roi)))
        }
        
        return features

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
        Train the network using gradient descent
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
        """
        for _ in range(self.random_walk_iterations):
            # Perform random walk (gradient descent with some randomness)
            gradient_weights = np.zeros_like(self.weights)
            gradient_bias = 0
            
            for i in range(len(X)):
                prediction = self.sigmoid(np.dot(X[i], self.weights) + self.bias)
                error = y[i] - prediction
                
                # Add some randomness to gradient
                noise_weights = np.random.normal(0, 0.1, self.weights.shape)
                noise_bias = np.random.normal(0, 0.1)
                
                gradient_weights += error * X[i] + noise_weights
                gradient_bias += error + noise_bias
            
            # Update weights and bias
            self.weights += self.learning_rate * gradient_weights / len(X)
            self.bias += self.learning_rate * gradient_bias / len(X)
    
    def random_walk_classification(self, graph, start_node, max_steps=10):
        """
        Perform random walk on the network graph for classification
        
        Args:
            graph (nx.Graph): Network graph
            start_node (int): Starting node for random walk
            max_steps (int): Maximum number of steps in random walk
        
        Returns:
            int: Predicted class after random walk
        """
        current_node = start_node
        node_features = []
        
        for _ in range(max_steps):
            # Collect features from current node
            node_features.append(graph.nodes[current_node]['features'])
            
            # Move to a random neighbor
            neighbors = list(graph.neighbors(current_node))
            if not neighbors:
                break
            
            current_node = random.choice(neighbors)
        
        # Aggregate features and predict
        if node_features:
            aggregated_features = np.mean(node_features, axis=0)
            return self.predict(aggregated_features)
        
        return None

def prepare_dataset(roi_dataset):
    """
    Prepare dataset for network training
    
    Args:
        roi_dataset (BladderCancerROIDataset): ROI dataset
    
    Returns:
        tuple: Prepared features, labels, and graph
    """
    # Extract features and labels
    features = []
    labels = []
    
    # Create a graph to represent relationships between ROIs
    G = nx.Graph()
    
    for idx in range(len(roi_dataset)):
        sample = roi_dataset[idx]
        roi = sample['image']
        
        # Extract features
        feature_dict = FeatureExtractor.extract_features(roi)
        feature_vector = np.array(list(feature_dict.values()))
        
        # Determine label based on case type
        label = 1 if sample['case_type'] == 'Lesion' else 0
        
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

def main(roi_dataset):
    """
    Main function to train and evaluate the rule-based network
    
    Args:
        roi_dataset (BladderCancerROIDataset): ROI dataset
    """
    # Prepare dataset
    features, labels, graph = prepare_dataset(roi_dataset)
    
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
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Perform random walk classification on graph
    print("\nRandom Walk Classification Demonstration:")
    for _ in range(5):  # Try 5 random walks
        start_node = random.choice(list(graph.nodes()))
        walk_prediction = network.random_walk_classification(graph, start_node)
        print(f"Random Walk from Node {start_node}: Predicted Class = {walk_prediction}")

# base_dataset = BladderCancerDataset(root_dir='data/preprocessed/Al-Bladder Cancer/')
# roi_dataset = BladderCancerROIDataset(
#     base_dataset, 
#     roi_width=128, 
#     roi_height=128, 
#     overlap=0.40, 
#     max_rois_per_image=10
# )
# main(roi_dataset)