import gymnasium as gym
from gymnasium import spaces
import numpy as np
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt
from minorminer import find_embedding
import networkx as nx
import dwave_networkx as dnx
import math

class GraphEmbEnv(gym.Env):

    def __init__(self, source_graph_set, target_graph, delta_heat, norm):
        super(GraphEmbEnv, self).__init__()
        
        self.source_graph_set = source_graph_set
        self.curr_graph = 0
        self.target_graph = target_graph
        self.delta_heat = delta_heat
        self.aux_nodes = {}
        self.embeddings = {}
        self.nodes_heat = {}
        self.norm = norm
        self.max_heat = 0
        self.priority_list = []
        self.priority_node = None
        self.invalid_ep = False
        self.max_conn = 10
        self.count_steps = 0
        high_obs = 1 if norm else 1000

        self.observation_space = spaces.Box(low=0, high=high_obs, shape=(self.max_conn,), dtype=np.float32)

        self.action_space = spaces.Discrete(self.max_conn)

        self.curr_graph = (self.curr_graph+1) % len(self.source_graph_set)
        self.source_graph = self.source_graph_set[self.curr_graph]
        self.modified_graph = self.source_graph.copy()

        self.find_local_embeddings(hops=1)
        self.nodes_heat = self.heat_function()
        self.start_heat = self.get_avg_heat()
        #-print("START HEAT ", self.embeddings)
        self.target_heat = self.get_target_heat()

        self.state = np.array(self.adjust_state_size(self.state_function()), dtype=np.float32)

    def step(self, action):

        aux_to_remove = set().union(*self.aux_nodes.values())
        neighbors = sorted(self.modified_graph.neighbors(self.priority_node))
        neighbors = [elem for elem in neighbors if elem not in aux_to_remove]
        #-print("NEIGHBORS")
        #-print(neighbors)
        reward = 0
        avail_aux_node = None
        
        if(action < len(neighbors)):
            avail_aux_node = self.get_avail_aux_node()
            if(avail_aux_node==None):
                avail_aux_node = self.add_aux_node()

            #-print("NEIGH ACT")
            #-print(neighbors[action])
            self.modified_graph.remove_edge(self.priority_node, neighbors[action])
            self.modified_graph.add_edge(avail_aux_node, neighbors[action])
            
            prev_avg_heat = self.get_avg_heat()
            update_nodes_list = [self.priority_node, neighbors[action], avail_aux_node]
            #-print("UPDATE")
            #-print(update_nodes_list)
            self.find_local_embeddings(hops=1, subset=update_nodes_list)
            self.update_heat_function(subset=update_nodes_list)

            # Calcola il reward
            reward = self.reward_function(prev_avg_heat)

        else:

            case_complexity = self.get_case_complexity()
            reward = (-2)

        #print("MOD GRAPH {}".format(self.modified_graph.edges()))

        self.count_steps = self.count_steps + 1

        # Verifica la condizione di terminazione
        terminated = self.check_terminated()

        info = self.get_info(reward, self.state, action, neighbors, avail_aux_node)

        self.state = np.array(self.adjust_state_size(self.state_function()), dtype=np.float32)

        #no truncated?

        return self.state, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        self.embeddings = {}
        self.aux_nodes = {}
        self.priority_list = []
        self.priority_node = None
        self.invalid_ep = False
        self.count_steps = 0
        self.max_heat = 0

        self.curr_graph = (self.curr_graph+1) % len(self.source_graph_set)
        self.source_graph = self.source_graph_set[self.curr_graph]
        self.modified_graph = self.source_graph.copy()
        #print(f"### NOW ON GRAPH {self.curr_graph} ###")

        self.find_local_embeddings(hops=1)
        self.nodes_heat = self.heat_function()
        self.start_heat = self.get_avg_heat()
        #-print("START HEAT ", self.nodes_heat)
        #-print("START EMBEDDING ", self.embeddings)
        self.target_heat = self.get_target_heat()

        info = self.get_info(None, self.state, np.array(-1), [], -1)

        self.state = np.array(self.adjust_state_size(self.state_function()), dtype=np.float32)

        return self.state, info
    
    def reward_function(self, prev_avg_heat):
        opposite_improvement = math.log(self.get_avg_heat()/prev_avg_heat)
        case_complexity = self.get_case_complexity()
        reward = -opposite_improvement
        return reward

    def get_subgraph_by_hop(self, node, hops):
        H = self.modified_graph
        nodes_in_hop = {node}
        for _ in range(hops):
            neighbors = set()
            for n in nodes_in_hop:
                #print(n)
                #print(list(H.neighbors(n)))
                neighbors.update(H.neighbors(n))
            nodes_in_hop.update(neighbors)
        #print(nodes_in_hop)
        return H.subgraph(nodes_in_hop)
    
    def find_local_embeddings(self, hops, subset=None):

        nodes_list = []
        if not subset:
            #-print("IF")
            H = self.modified_graph
            nodes_list = list(H.nodes())
        else:
            #-print("ELSE")
            nodes_list = subset

        #-print(nodes_list)

        #H_og_nodes = set(H.nodes()).difference(set(self.aux_nodes))

        for node in nodes_list:
            H_sub = self.get_subgraph_by_hop(node, hops)
            embedding = False
            dim = 3
            
            while not embedding:
                if hops==1:
                    chimera_graph = dnx.chimera_graph(dim, dim, 4)
                else:
                    chimera_graph = dnx.chimera_graph(12, 12, 4)
                embedding = find_embedding(H_sub, chimera_graph)
                dim = dim+1
            if not embedding:
                print("AH! ", dim)
            self.embeddings[node] = embedding
    
    def heat_function(self):
        #H_og_nodes = set(self.modified_graph.nodes()).difference(set(self.aux_nodes))
        H = self.modified_graph

        nodes_heat = {}

        for node in H.nodes():
            nodes_heat[node] = 0

        for viewpoint in self.embeddings:
            local_embedding = self.embeddings[viewpoint]
            vertex_model = viewpoint
            nodes_heat[vertex_model] = len(local_embedding[vertex_model])
            self.max_heat = nodes_heat[vertex_model] if nodes_heat[vertex_model] > self.max_heat else self.max_heat
            #-print("MAX HEAT ", self.max_heat)

        if(self.norm):
            for node in nodes_heat:
                nodes_heat[node] = nodes_heat[node]/self.max_heat
        else:
            for node in nodes_heat:
                nodes_heat[node] = nodes_heat[node]

        return nodes_heat
    
    def update_heat_function(self, subset):
        for node in subset:
            local_embedding = self.embeddings[node]
            node_heat = len(local_embedding[node])
            self.max_heat = node_heat if node_heat > self.max_heat else self.max_heat
            if(self.norm):
                self.nodes_heat[node] = node_heat/self.max_heat
            else:
                self.nodes_heat[node] = node_heat
            
    
    def state_function(self):
       
        self.priority_list = [key for key, value in sorted(self.nodes_heat.items(), key=lambda item: item[1], reverse=True)]
        #-print("PRIORITA")
        #-print(self.priority_list)
        aux_to_remove = set().union(*self.aux_nodes.values())
        self.priority_list = [elem for elem in self.priority_list if elem not in aux_to_remove]
        #-print("PRIORITA DOPO")
        #-print(self.priority_list)
        self.priority_node = self.priority_list[0]

        neighbors = set()
        neighbors.update(self.modified_graph.neighbors(self.priority_node))

        #print(self.nodes_heat)
        #print(sorted(neighbors))

        return [self.nodes_heat[neighbor] for neighbor in sorted(neighbors)]
    
    def adjust_state_size(self, state_heat):
        if len(state_heat) > self.max_conn:
            return state_heat[:self.max_conn]
        elif len(state_heat) < self.max_conn:
            return state_heat + [0] * (self.max_conn - len(state_heat))
        return state_heat
    
    def get_avail_aux_node(self):
        if self.priority_node in self.aux_nodes:
            local_aux_nodes = self.aux_nodes[self.priority_node]
            min_heat = 1 if self.norm else 1000
            min_heat_node = -1
            for node in local_aux_nodes:
                min_heat, min_heat_node = (self.nodes_heat[node], node) if self.nodes_heat[node] < min_heat else (min_heat, min_heat_node)
            if(min_heat < self.target_heat):
                    return min_heat_node
        return None

    def add_aux_node(self):
        aux_node = len(self.modified_graph.nodes())+1
        n_priority_node_neighbors = len(list(self.modified_graph.neighbors(self.priority_node)))
        n_aux_nodes_neighbors = len(self.aux_nodes[self.priority_node]) if self.priority_node in self.aux_nodes else 0
        n_og_nodes_neighbors = n_priority_node_neighbors - n_aux_nodes_neighbors
        
        self.modified_graph.add_node(aux_node)

        #if n of aux nodes is smaller than the original neighbors left of the priority node / 2 than
        #add an aux node directly to the priority node, else add it to the aux node with lower heat
        if(n_aux_nodes_neighbors < math.ceil(n_og_nodes_neighbors/2)):
            self.modified_graph.add_edge(self.priority_node, aux_node)
        else:
            local_aux_nodes = self.aux_nodes[self.priority_node]
            min_heat = 1 if self.norm else 1000
            min_heat_node = -1
            for node in local_aux_nodes:
                min_heat, min_heat_node = (self.nodes_heat[node], node) if self.nodes_heat[node] < min_heat else (min_heat, min_heat_node)
            self.modified_graph.add_edge(min_heat_node, aux_node)
        if(self.priority_node in self.aux_nodes):
            self.aux_nodes[self.priority_node].update(set([aux_node]))
        else:
            self.aux_nodes[self.priority_node] = set([aux_node])
            
        return aux_node
    
    def remove_aux_nodes(self):
        p_list = self.priority_list
        
        for node in self.aux_nodes:
            p_list = list(set(p_list).difference(self.aux_nodes[node]))
        return p_list

    def get_avg_heat(self):
        avg_heat = 0
        for node in self.nodes_heat:
            avg_heat = avg_heat + self.nodes_heat[node]
        return avg_heat/len(self.nodes_heat)

    #così scala in base all'avg_heat, prova anche versione fissa, avg_heat-delta_heat
    def get_target_heat(self):
        avg_heat = self.get_avg_heat()
        return avg_heat*(1-self.delta_heat)
    
    def get_case_complexity(self, alpha=1.0, beta=1.5, gamma=1.0, delta=1.2):
        complexity = alpha * (len(self.source_graph.nodes()) ** gamma) + beta * (len(self.source_graph.edges()) ** delta)
        return complexity
    
    def update_source_graph(self, new_source_graph):
        self.source_graph = new_source_graph
        self.reset()
    
    def get_info(self, reward, state, action, neighbors, avail_aux_node):
        info = {"start_heat": self.start_heat,
                "avg_heat": self.get_avg_heat(),
                "target_heat": self.target_heat,
                "nodes_heat": self.nodes_heat.copy(),
                "priority_node": self.priority_node,
                "embeddings": self.embeddings.copy(),
                "reward": reward,
                "state": state.tolist().copy(),
                "action": action.item(),
                "neighbors": neighbors.copy(),
                "avail_aux_node": avail_aux_node,
                "modified_graph": self.modified_graph.copy(),
                "aux_nodes": self.aux_nodes.copy(),
                "invalid_ep": self.invalid_ep}
        return info

    def check_terminated(self):
        # Verifica se tutti i nodi del grafo di input sono stati mappati o se sono finiti i nodi target
        if self.count_steps > len(self.source_graph.nodes()):
            self.invalid_ep = True
            print("INVALID EP!")
        return (self.get_avg_heat() <= self.target_heat) or (self.count_steps > len(self.source_graph.nodes())) #cambia condizione counter per grafi GRANDI
