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
# no_of_appearances_per_node_dict=dict.fromkeys(list(G.nodes), 0)
# no_of_appearances_per_node_dict[next_node]=1


## ======================== RANDOM WALK: ========================
path=[next_node]
max_walk_length=100

for i in range(max_walk_length):
    neighbors = list(G.neighbors(next_node))
    if not neighbors:
        nodes_without_neighbors.append(next_node)
        next_node = random.choice(set(nodes).difference(set(nodes_without_neighbors)))
        continue
    else:
        random_node = random.choice(list_for_nodes) #choose a node randomly from neighbors
        dict_counter[random_node] = dict_counter[random_node]+1

# ========== Generate, save and show final plot (fig1): ==========
fig, axes = plt.subplot_mosaic(layout, figsize=(25,25))
ax=nx.draw(G, ax=axes["orig_network"], node_color="yellow", edge_color="black", with_labels=True, font_weight="bold") #draw the network graph 
ax=nx.draw(G, ax=axes["rw_network"], node_color="green", edge_color="black", with_labels=True, font_weight="bold") #draw the network graph 
# Plot settings: 
# plt.subplots_adjust(wspace=0.45, hspace=3.3)
plt.savefig('rw-network.pdf', format='pdf', bbox_inches='tight')
# fig.tight_layout(pad=100.0)
plt.show()