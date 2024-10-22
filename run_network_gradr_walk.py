import networkx as nx
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
from Network_GradR_Walk.network_gradr_walk import network_gradr_walk
import random

L=100 # max_walk_length_limit
N=1000 # max_no_of_random_walks_limit
mode=1 # 1: Repetition of nodes allowed, 2: No repetition of nodes, 3: Repetition of nodes only allowed when no unvisited neighbors remains                         
gname="G_rw" # save_random_walk_graph_as

layout = [
    ["orig_network"],
    ["rw_network"]
]

# Dummy network for testing:
G = nx.complete_graph(10)
attributes_dummy_list=[]
no_of_elements=len(list(G.nodes))
for element in range(no_of_elements):
    attributes_dummy_list.append(random.randint(0,10))
attributes_dummy_list = [x/float(10) for x in attributes_dummy_list]
attributes_dict=dict(zip(list(G.nodes), attributes_dummy_list))
nx.set_node_attributes(G, attributes_dict, "attr1")

attr="attr1" # attribute_name

list_of_random_walk_graphs=network_gradr_walk(G, max_walk_length_limit=L, max_no_of_random_walks_limit=N, mode=mode, save_random_walk_graph_as=gname)

# # ========== Generate, save and show final plot (fig1): ==========
# fig, axes = plt.subplot_mosaic(layout, figsize=(25,25))
# ax=nx.draw(G, ax=axes["orig_network"], node_color="yellow", edge_color="black", with_labels=True, font_weight="bold") #draw the network graph 
# ax=nx.draw(G_rw, ax=axes["rw_network"], node_color="green", edge_color="black", with_labels=True, font_weight="bold") #draw the network graph 
# # Plot settings: 
# # plt.subplots_adjust(wspace=0.45, hspace=3.3)
# plt.savefig('rw-network.pdf', format='pdf', bbox_inches='tight')
# # fig.tight_layout(pad=100.0)
# plt.show()