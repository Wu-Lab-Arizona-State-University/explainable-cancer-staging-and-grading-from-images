import numpy as np
import networkx as nx
import random
from Dataloader import BladderCancerDataset, BladderCancerROIDataset

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

# Random Walk Explanation
RANDOM_WALK_EXPLANATION = """
Random Walk Classification Algorithm Explanation:

1. Core Concept:
   - Start at a specific node in the graph
   - Traverse through connected nodes
   - Collect and aggregate features during traversal
   - Use network's predictive model to classify based on aggregated features

2. Key Components:
   a) Exploration vs Exploitation:
      - Controlled by exploration probability
      - Allows both following similar paths and discovering new routes
   
   b) Node Selection Strategy:
      - Uses cosine similarity between node features
      - Applies softmax to convert similarities to probabilistic node selection
      - Favors nodes with more similar features while maintaining randomness

3. Algorithmic Steps:
   - Initialize at start node
   - Collect node features during walk
   - Select next node based on feature similarity
   - Limit walk by maximum steps
   - Aggregate features from visited nodes
   - Predict class using network's prediction method

4. Advantages:
   - Captures local graph structure
   - Handles feature aggregation dynamically
   - Introduces controlled randomness
   - Adapts to graph's local feature landscape

5. Hyperparameters:
   - max_steps: Controls walk length
   - exploration_prob: Balances exploitation and exploration
"""

# Print the explanation
print(RANDOM_WALK_EXPLANATION)