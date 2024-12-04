import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from Dataloader import BladderCancerDataset,BladderCancerROIDataset

class ROINetworkRandomWalk:
    def __init__(self, roi_dataset):

        self.dataset = roi_dataset
        self.graph = self._construct_roi_graph()

    def _construct_roi_graph(self):
        """
        Construct a graph from ROI dataset
        
        Returns:
        --------
        networkx.DiGraph: Directed graph representing ROI relationships
        """
        G = nx.DiGraph()
        
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            
            roi = sample['image'].squeeze().numpy()
            
            node_attributes = {
                'attr1': np.mean(roi),  
                'time_point': sample['time_point'],
                'ct_folder': sample['ct_folder'],
                'case_type': sample['case_type'],
                'original_index': idx
            }
            
            G.add_node(idx, **node_attributes)
        
        for i in range(len(self.dataset)):
            for j in range(len(self.dataset)):
                if i != j:
                    if (self.graph.nodes[i]['time_point'] == self.graph.nodes[j]['time_point'] and 
                        self.graph.nodes[i]['case_type'] == self.graph.nodes[j]['case_type']):
                        G.add_edge(i, j)
        
        return G

    def network_gradr_walk(self, 
                            max_walk_length_limit=100, 
                            max_no_of_random_walks_limit=1000, 
                            mode=1):
        """
         original network_gradr_walk function
        """
        attr = nx.get_node_attributes(self.graph, "attr1")
        nodes = list(self.graph.nodes)
        
        random_walk_graphs = []
        no_of_random_walks = 0
        
        while no_of_random_walks <= max_no_of_random_walks_limit:
            next_node = random.choice(nodes)
            
            attribute_list = [attr[next_node]]
            path = [next_node]
            path_description = []
            no_of_appearances_per_node_dict = dict.fromkeys(nodes, 0)
            no_of_appearances_per_node_dict[next_node] = 1
            
            for _ in range(max_walk_length_limit):
                neighbors = list(self.graph.neighbors(next_node))
                
                if not neighbors:
                    path_description.append('no_neighbors')
                    next_node = random.choice(nodes)
                    attribute_list.append(attr[next_node])
                    no_of_random_walks += 1
                    continue
                
                if mode == 1: 
                    gradient_scores = {}
                    for neighbor in neighbors:
                        gradient_scores[neighbor] = (
                            attr[neighbor] - attr[path[-1]]
                        ) / (attr[path[-1]] + 1e-8)  # Prevent division by zero
                    
                    max_val = max(gradient_scores.values())
                    best_neighbors = [
                        n for n, score in gradient_scores.items() 
                        if score == max_val
                    ]
                    
                    next_node = random.choice(best_neighbors)
                    path_description.append('_')
                    attribute_list.append(attr[next_node])
                    path.append(next_node)
                    no_of_appearances_per_node_dict[next_node] += 1
                
                elif mode == 2:  
                    neighbors_already_visited = set(neighbors).intersection(set(path))
                    neighbors_not_visited = set(neighbors).difference(neighbors_already_visited)
                    
                    if not neighbors_not_visited:
                        path_description.append('no_unvisited_neighbors')
                        next_node = random.choice(nodes)
                        attribute_list.append(attr[next_node])
                        no_of_random_walks += 1
                        continue
                    
                    gradient_scores = {}
                    for neighbor in neighbors_not_visited:
                        gradient_scores[neighbor] = (
                            attr[neighbor] - attr[path[-1]]
                        ) / (attr[path[-1]] + 1e-8)
                    
                    max_val = max(gradient_scores.values())
                    best_neighbors = [
                        n for n, score in gradient_scores.items() 
                        if score == max_val
                    ]
                    
                    next_node = random.choice(best_neighbors)
                    path_description.append('_')
                    attribute_list.append(attr[next_node])
                    path.append(next_node)
                    no_of_appearances_per_node_dict[next_node] += 1
                
                elif mode == 3: 
                    neighbors_already_visited = set(neighbors).intersection(set(path))
                    neighbors_not_visited = set(neighbors).difference(neighbors_already_visited)
                    
                    if not neighbors_not_visited:
                        path_description.append('no_unvisited_neighbors')
                        next_node = random.choice(list(neighbors_already_visited))
                        attribute_list.append(attr[next_node])
                        path.append(next_node)
                        no_of_appearances_per_node_dict[next_node] += 1
                    else:
                        gradient_scores = {}
                        for neighbor in neighbors_not_visited:
                            gradient_scores[neighbor] = (
                                attr[neighbor] - attr[path[-1]]
                            ) / (attr[path[-1]] + 1e-8)
                        
                        max_val = max(gradient_scores.values())
                        best_neighbors = [
                            n for n, score in gradient_scores.items() 
                            if score == max_val
                        ]
                        
                        next_node = random.choice(best_neighbors)
                        path_description.append('_')
                        attribute_list.append(attr[next_node])
                        path.append(next_node)
                        no_of_appearances_per_node_dict[next_node] += 1
            
            path_description_dict = dict(zip(path, path_description))
            G_rw = nx.DiGraph()
            nx.add_path(G_rw, path)
            nx.set_node_attributes(G_rw, path_description_dict, "path_description")
            random_walk_graphs.append(G_rw)
            
            no_of_random_walks += 1
        
        return random_walk_graphs

    def visualize_random_walks(self, random_walk_graphs, num_walks_to_plot=3):

        for walk_idx, G_walk in enumerate(random_walk_graphs[:num_walks_to_plot]):
            # Get the nodes in order
            walk_nodes = list(nx.topological_sort(G_walk))
            
            # Create figure for this walk
            plt.figure(figsize=(15, 5))
            
            for i, node in enumerate(walk_nodes):
                # Get original ROI sample
                sample = self.dataset[self.graph.nodes[node]['original_index']]
                roi = sample['image'].squeeze().numpy()
                
                # Plot ROI
                plt.subplot(1, len(walk_nodes), i+1)
                plt.imshow(roi, cmap='gray')
                plt.title(f"Step {i}\n{sample['time_point']}\n{sample['case_type']}")
                plt.axis('off')
            
            plt.suptitle(f'Random Walk {walk_idx+1}')
            plt.tight_layout()
            plt.show()

    def analyze_walks(self, random_walk_graphs):

        walk_lengths = [len(list(G.nodes)) for G in random_walk_graphs]
        
        time_point_transitions = {}
        case_type_transitions = {}
        
        for G_walk in random_walk_graphs:
            walk_nodes = list(nx.topological_sort(G_walk))
            
            for i in range(len(walk_nodes) - 1):
                node1 = walk_nodes[i]
                node2 = walk_nodes[i+1]
                
                tp1 = self.graph.nodes[node1]['time_point']
                tp2 = self.graph.nodes[node2]['time_point']
                time_point_key = (tp1, tp2)
                time_point_transitions[time_point_key] = time_point_transitions.get(time_point_key, 0) + 1
                
                ct1 = self.graph.nodes[node1]['case_type']
                ct2 = self.graph.nodes[node2]['case_type']
                case_type_key = (ct1, ct2)
                case_type_transitions[case_type_key] = case_type_transitions.get(case_type_key, 0) + 1
        
        return {
            'total_walks': len(random_walk_graphs),
            'walk_lengths': {
                'min': min(walk_lengths),
                'max': max(walk_lengths),
                'mean': np.mean(walk_lengths)
            },
            'time_point_transitions': time_point_transitions,
            'case_type_transitions': case_type_transitions
        }

# def run_roi_random_walk(base_dataset_path, roi_width=128, roi_height=128):
#     # Create base and ROI datasets
#     base_dataset = BladderCancerDataset(root_dir=base_dataset_path)
#     roi_dataset = BladderCancerROIDataset(
#         base_dataset, 
#         roi_width=roi_width, 
#         roi_height=roi_height, 
#         overlap=0.40, 
#         max_rois_per_image=10
#     )
    
#     # Initialize ROI Network Random Walk
#     roi_network_walk = ROINetworkRandomWalk(roi_dataset)
    
#     # Perform random walks
#     for mode in [1, 2, 3]:
#         print(f"\nRandom Walk Mode {mode}")
#         random_walk_graphs = roi_network_walk.network_gradr_walk(
#             max_walk_length_limit=10,
#             max_no_of_random_walks_limit=50,
#             mode=mode
#         )
        
#         # Visualize walks
#         roi_network_walk.visualize_random_walks(random_walk_graphs)
        
#         # Analyze walks
#         walk_analysis = roi_network_walk.analyze_walks(random_walk_graphs)
#         print(walk_analysis)
        
# run_roi_random_walk(base_dataset_path='/home/as-aravinthakshan/Desktop/RESEARCH/explainable-cancer-staging-and-frading-from-images/data/preprocessed/Al-Bladder Cancer')