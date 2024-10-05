import networkx as nx
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
import random

layout = [
    ["orig_network"],
    ["rw_network"]
]


G = nx.complete_graph(10)
nodes=list(G.nodes)
next_node = random.choice(nodes)
nodes_without_neighbors=[]
no_of_appearances_per_node_dict=dict.fromkeys(list(G.nodes), 0)
no_of_appearances_per_node_dict[next_node]=1


## ======================== RANDOM WALK: ========================
path=[next_node]
path_description=[]
max_walk_length_limit=100
max_no_of_random_walks_limit=1000
no_of_random_walks=0

# 1. Repetition of nodes allowed:
while(no_of_random_walks<=max_no_of_random_walks_limit):
    for i in range(max_walk_length_limit):
        neighbors = list(G.neighbors(next_node))
        if not neighbors:
            nodes_without_neighbors.append(next_node)
            next_node = random.choice(set(nodes).difference(set(nodes_without_neighbors)))
            no_of_random_walks+=1
            continue
        else:
            next_node = random.choice(neighbors) #choose a node randomly from neighbors
            path.append(next_node)
            no_of_appearances_per_node_dict[next_node] = no_of_appearances_per_node_dict[next_node]+1

# 2. No repetition of nodes:
while(no_of_random_walks<=max_no_of_random_walks_limit):
    for i in range(max_walk_length_limit):
        neighbors = list(G.neighbors(next_node))
        if not neighbors:
            path_description.append('no_neighbors')
            nodes_without_neighbors.append(next_node)
            next_node = random.choice(set(nodes).difference(set(nodes_without_neighbors)))
            no_of_random_walks+=1
            continue
        else:
            neighbors_already_visited=set(neighbors).intersection(set(path))
            neighbors_not_visited=set(neighbors).difference(set(neighbors_already_visited))
            if not neighbors_not_visited:
                path_description.append('no_unvisited_neighbors')
                next_node = random.choice(set(nodes).difference(set(nodes_without_neighbors)))
                no_of_random_walks+=1
                continue
            
            else:
                next_node = random.choice(neighbors_not_visited) #choose a node randomly from neighbors
                path.append(next_node)
                no_of_appearances_per_node_dict[next_node] = no_of_appearances_per_node_dict[next_node]+1

# 3. Repetition of nodes only allowed when no unvisited neighbors remains:
while(no_of_random_walks<=max_no_of_random_walks_limit):
    for i in range(max_walk_length_limit):
        neighbors = list(G.neighbors(next_node))
        if not neighbors:
            path_description.append('no_neighbors')
            nodes_without_neighbors.append(next_node)
            next_node = random.choice(set(nodes).difference(set(nodes_without_neighbors)))
            no_of_random_walks+=1
            continue
        else:
            neighbors_already_visited=set(neighbors).intersection(set(path))
            neighbors_not_visited=set(neighbors).difference(set(neighbors_already_visited))
            if not neighbors_not_visited:
                path_description.append('no_unvisited_neighbors')
                next_node = random.choice(neighbors_already_visited)
                path.append(next_node)
                no_of_appearances_per_node_dict[next_node] = no_of_appearances_per_node_dict[next_node]+1
            else:
                next_node = random.choice(neighbors_not_visited) #choose a node randomly from neighbors
                path.append(next_node)
                no_of_appearances_per_node_dict[next_node] = no_of_appearances_per_node_dict[next_node]+1

# ========== Generate network with random walk ==================
G_rw = nx.DiGraph()  # or MultiDiGraph, etc
nx.add_path(G_rw, path)

# ========== Generate, save and show final plot (fig1): ==========
fig, axes = plt.subplot_mosaic(layout, figsize=(25,25))
ax=nx.draw(G, ax=axes["orig_network"], node_color="yellow", edge_color="black", with_labels=True, font_weight="bold") #draw the network graph 
ax=nx.draw(G_rw, ax=axes["rw_network"], node_color="green", edge_color="black", with_labels=True, font_weight="bold") #draw the network graph 
# Plot settings: 
# plt.subplots_adjust(wspace=0.45, hspace=3.3)
plt.savefig('rw-network.pdf', format='pdf', bbox_inches='tight')
# fig.tight_layout(pad=100.0)
plt.show()