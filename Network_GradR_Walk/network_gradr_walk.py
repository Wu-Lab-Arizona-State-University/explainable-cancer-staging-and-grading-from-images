import random
import networkx as nx

def network_gradr_walk(G, max_walk_length_limit=100, max_no_of_random_walks_limit=1000, mode=1, gname="G_rw", attribute_name="attr1"):
    """Returns list of directed random walk graphs

    Parameters
    ----------
    p: Initial population # Np = len(p) : # of individuals in population during initialization (or, # of chromosomes - one chromosome per individual) [Note that $Np$ is different from actual population $p = list_of_chromosomes_during_initialization$] 
    
    Np_cap: Population cap at every step barring initialization, i.e., # of stronger individuals to be retained at the end of each step.
    
    alpha : rate of mutation
    
    goal : goal list (goal sequence) - here, embedding from cancer ROI
    
    g: set of all permissible values that a particular gene can assume.
    
    copy : bool (Optional parameter | default False)
        $copy = True$ returns list of lists, and all scores.
        $copy = False$ saves list of lists, and all scores.
    
    Returns
    -------
    i) list of lists, ii) scores
    
    Note
    ----
    Size of a chromosome (that is, number of genes contained in a chromosome) is constant across all individuals.

    """
    attr = nx.get_node_attributes(G, "attr1")
    nodes=list(G.nodes)
    isolates=nx.isolates(G)

    ## ======================== RANDOM WALK: ========================
    random_walk_graphs=[]
    no_of_random_walks=0
    
    if mode==1:
        # 1. Repetition of nodes allowed:
        while(no_of_random_walks<=max_no_of_random_walks_limit):
            attribute_list=[]
            next_node = random.choice(nodes)
            attribute_list.append(attr[next_node])
            nodes_without_neighbors=[]
            no_of_appearances_per_node_dict=dict.fromkeys(list(G.nodes), 0)
            no_of_appearances_per_node_dict[next_node]=1
            path=[next_node]
            path_description=[]
            for i in range(max_walk_length_limit):
                neighbors = list(G.neighbors(next_node))
                if not neighbors:
                    path_description.append('no_neighbors')
                    nodes_without_neighbors.append(next_node)
                    next_node = random.choice(set(nodes).difference(set(nodes_without_neighbors)))
                    attribute_list.append(attr[next_node])
                    no_of_random_walks+=1
                    continue
                else:
                    neighbor_attributes={}
                    gradient_scores={}
                    for neighbor in neighbors:
                        neighbor_attributes[neighbor]=nx.get_node_attributes(G, attribute_name)[neighbor]
                        gradient_scores[neighbor]=(neighbor_attributes[neighbor]-nx.get_node_attributes(G, attribute_name)[path[-1]])/nx.get_node_attributes(G, attribute_name)[path[-1]]
                    max_val = max(gradient_scores.values())
                    res = list(filter(lambda x: gradient_scores[x] == max_val, gradient_scores))
                    path_description.append('_')
                    next_node = random.choice(res) # neighbors #choose a node randomly from neighbors
                    attribute_list.append(attr[next_node])
                    path.append(next_node)
                    no_of_appearances_per_node_dict[next_node] = no_of_appearances_per_node_dict[next_node]+1
            path_description_dict=dict(zip(path, path_description))
            G_rw = nx.DiGraph()  # or MultiDiGraph, etc
            nx.add_path(G_rw, path)
            nx.set_node_attributes(G_rw, path_description_dict, "path_description")
            random_walk_graphs.append(G_rw)
    if mode==2:
        # 2. No repetition of nodes:
        while(no_of_random_walks<=max_no_of_random_walks_limit):
            attribute_list=[]
            next_node = random.choice(nodes)
            attribute_list.append(attr[next_node])
            nodes_without_neighbors=[]
            no_of_appearances_per_node_dict=dict.fromkeys(list(G.nodes), 0)
            no_of_appearances_per_node_dict[next_node]=1
            path=[next_node]
            path_description=[]
            for i in range(max_walk_length_limit):
                neighbors = list(G.neighbors(next_node))
                if not neighbors:
                    path_description.append('no_neighbors')
                    nodes_without_neighbors.append(next_node)
                    next_node = random.choice(set(nodes).difference(set(nodes_without_neighbors)))
                    attribute_list.append(attr[next_node])
                    no_of_random_walks+=1
                    continue
                else:
                    neighbors_already_visited=set(neighbors).intersection(set(path))
                    neighbors_not_visited=set(neighbors).difference(set(neighbors_already_visited))
                    if not neighbors_not_visited:
                        path_description.append('no_unvisited_neighbors')
                        next_node = random.choice(set(nodes).difference(set(nodes_without_neighbors)))
                        attribute_list.append(attr[next_node])
                        no_of_random_walks+=1
                        continue
                    else:
                        neighbor_attributes={}
                        gradient_scores={}
                        for neighbor in neighbors_not_visited:
                            neighbor_attributes[neighbor]=nx.get_node_attributes(G, attribute_name)[neighbor]
                            gradient_scores[neighbor]=(neighbor_attributes[neighbor]-nx.get_node_attributes(G, attribute_name)[path[-1]])/nx.get_node_attributes(G, attribute_name)[path[-1]]
                        max_val = max(gradient_scores.values())
                        res = list(filter(lambda x: gradient_scores[x] == max_val, gradient_scores))
                        path_description.append('_')
                        next_node = random.choice(res) # neighbors_not_visited #choose a node randomly from neighbors
                        attribute_list.append(attr[next_node])
                        path.append(next_node)
                        no_of_appearances_per_node_dict[next_node] = no_of_appearances_per_node_dict[next_node]+1
            path_description_dict=dict(zip(path, path_description))
            G_rw = nx.DiGraph()  # or MultiDiGraph, etc
            nx.add_path(G_rw, path)
            nx.set_node_attributes(G_rw, path_description_dict, "path_description")
            random_walk_graphs.append(G_rw)
    if mode==3:
        # 3. Repetition of nodes only allowed when no unvisited neighbors remains:
        while(no_of_random_walks<=max_no_of_random_walks_limit):
            attribute_list=[]
            next_node = random.choice(nodes)
            attribute_list.append(attr[next_node])
            nodes_without_neighbors=[]
            no_of_appearances_per_node_dict=dict.fromkeys(list(G.nodes), 0)
            no_of_appearances_per_node_dict[next_node]=1
            path=[next_node]
            path_description=[]
            for i in range(max_walk_length_limit):
                neighbors = list(G.neighbors(next_node))
                if not neighbors:
                    path_description.append('no_neighbors')
                    nodes_without_neighbors.append(next_node)
                    next_node = random.choice(set(nodes).difference(set(nodes_without_neighbors)))
                    attribute_list.append(attr[next_node])
                    no_of_random_walks+=1
                    continue
                else:
                    neighbors_already_visited=set(neighbors).intersection(set(path))
                    neighbors_not_visited=set(neighbors).difference(set(neighbors_already_visited))
                    if not neighbors_not_visited:
                        path_description.append('no_unvisited_neighbors')
                        next_node = random.choice(neighbors_already_visited)
                        attribute_list.append(attr[next_node])
                        path.append(next_node)
                        no_of_appearances_per_node_dict[next_node] = no_of_appearances_per_node_dict[next_node]+1
                    else:
                        neighbor_attributes={}
                        gradient_scores={}
                        for neighbor in neighbors_not_visited:
                            neighbor_attributes[neighbor]=nx.get_node_attributes(G, attribute_name)[neighbor]
                            gradient_scores[neighbor]=(neighbor_attributes[neighbor]-nx.get_node_attributes(G, attribute_name)[path[-1]])/nx.get_node_attributes(G, attribute_name)[path[-1]]
                        max_val = max(gradient_scores.values())
                        res = list(filter(lambda x: gradient_scores[x] == max_val, gradient_scores))
                        path_description.append('_')
                        next_node = random.choice(res) # neighbors_not_visited #choose a node randomly from neighbors
                        attribute_list.append(attr[next_node])
                        path.append(next_node)
                        no_of_appearances_per_node_dict[next_node] = no_of_appearances_per_node_dict[next_node]+1
            path_description_dict=dict(zip(path, path_description))
            G_rw = nx.DiGraph()  # or MultiDiGraph, etc
            nx.add_path(G_rw, path)
            nx.set_node_attributes(G_rw, path_description_dict, "path_description")
            random_walk_graphs.append(G_rw)
    # ========== Generate network with random walk ==================
    return random_walk_graphs
