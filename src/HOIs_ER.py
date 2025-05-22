import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.colors
import networkx as nx
import itertools
from itertools import combinations
from collections import defaultdict
from math import comb
import pickle
import time, os

#Here we create a hypernetwork with normal links and hyperlinks (triangles in particular) and the proportion of each is defined by alpha. Then in the dynamics there is no need of alpha, just pick a random link which can be pairwise or HO

os.chdir(r'C:\Users\mduran\Desktop\MSc\TFM\renewed_model')


H = np.matrix([[0.5, 0.34, 0.76], [0.66,0.5,0.25], [0.24,0.75,0.5]]) #interaction matrix


def generate_network_mixed(N,p1,p2,alpha): 
    
    while True:
        
        ## - FIRST PHASE
        ## - I first generate a standard ER graph with pairwise links connected with probability p1
        
        g_pairwise = nx.gnp_random_graph(N, p1)
        
        #Check pairwise giant component system
        giant = max(nx.connected_components(g_pairwise), key=len) #gc solo del sistema pairwise
        g_pairwise = g_pairwise.subgraph(giant).copy()
        
        # If I need a minimum pairwise giant component size
        '''
        while len(g_pairwise.nodes()) < N*0.95:
            
            print('FAIL - pairwise connected nodes',len(g_pairwise.nodes()))
            
            g_pairwise = nx.gnp_random_graph(N, p1)
            if not nx.is_connected(g_pairwise):
                giant = max(nx.connected_components(g_pairwise), key=len)
                g_pairwise = g_pairwise.subgraph(giant).copy()
        '''
        print('pairwise connected nodes', len(g_pairwise.nodes()))
        print('number of pairwise interactions', 2*g_pairwise.size())
        
        
        ## - SECOND PHASE
        ## - Now I run over all the possible combinations of three elements/nodes:
            
        triangles_list = []
        
        for tri in combinations([n for n in range(N)],3): #If I want triangles to be created among all possible triangle combinations within N
       
            #And I create the triangle with probability p2
            if random.random() < p2:
            
                triangles_list.append(tri)
        
        #print(triangles_list)
        
        
        #Check HOIs giant component system
        
        edges_triangles = []
        for triangle in triangles_list:
            edges_triangles.extend(list(itertools.combinations(triangle, 2)))
        #print(edges_triangles)
        g_triangles = nx.from_edgelist(edges_triangles)
        
        giant = max(nx.connected_components(g_triangles), key=len) #gc solo del sistema HO
        g_triangles = g_triangles.subgraph(giant).copy()
       
        # If I need a minimum HO giant component size
        '''
        while len(g_triangles.nodes()) < 1:
            
            print('FAIL - triangle connected nodes',len(g_triangles.nodes()))
            
            triangles_list = []
            for tri in combinations([n for n in range(N)],3):
           
                #And I create the triangle with probability p2
                if random.random() < p2:
                
                    triangles_list.append(tri)
                    
            #print('number of HOIs', 3*len(triangles_list))
            
            edges_triangles = []
            for triangle in triangles_list:
                edges_triangles.extend(list(itertools.combinations(triangle, 2)))
            #print(edges_triangles)
            g_triangles = nx.from_edgelist(edges_triangles)
            
            giant = max(nx.connected_components(g_triangles), key=len) #gc solo del sistema pairwise
            g_triangles = g_triangles.subgraph(giant).copy()
        '''
        
        print('triangle connected nodes',len(g_triangles.nodes()))
        print('number of HO interactions', 3*len(triangles_list))
       
        
        ## - THIRD PHASE
        ## - Combine HOIs and pairwise networks (both together)
        
        edges = []
        for triangle in triangles_list:
            edges.extend(list(itertools.combinations(triangle, 2)))
    
        for pairwise in g_pairwise.edges():
            edges.append(pairwise)
        
        G = nx.from_edgelist(edges) # Generate a network of both pairwise and HOIs
        giant = max(nx.connected_components(G), key=len) #gc del sistema pairwise i HOIs together
        G = G.subgraph(giant).copy()
        #print('total connected nodes', len(G.nodes()))
        
        
        # If I need a minimum combined pairwise and HO giant component size
        giant_component = max(nx.connected_components(G), key=len)
        G = G.subgraph(giant_component).copy()
        
        if len(G.nodes()) < N*0.95:
            
            print('FAIL - COMBINED connected nodes',len(G.nodes())) 
            
            # Regenerate the network by continuing the loop
            continue
    
        print('Total connected nodes in combined network:', len(G.nodes()))
        print('Number of combined interactions:', G.number_of_edges())
        
            
        
        # Initialize neighs for all nodes in G
        neighs = {}
        for node in g_pairwise.nodes():
            if node in G.nodes():
                # Neighbours of the current node in the pairwise network
                node_neighbours = [n for n in g_pairwise.neighbors(node) if n in G.nodes()]
                neighs[node] = node_neighbours
        
        #print(neighs) 
        
        #Save the network
                 
        f1 = open("triangles" + str(alpha) +  "_ER.txt", "w") 
        for i in triangles_list:
            #print(i)
            #print(i[0])
            f1.write( str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + "\n")
        #print(triangles_list)  
        f2 = open("pairwise_neighs" + str(alpha) +  "_ER.txt", "w") 
        for i in neighs:        
            #print(neighs[i])
            for j in range(len(neighs[i])):
                #print(neighs[i][j])
                f2.write( str(neighs[i][j]) + ' ' )
            f2.write('\n')
        #print(neighs)
            
        
        
        # Adding the HOIs to the neighbor dictionary
        for tri in triangles_list:
            # Only consider triangles fully contained in G
            if all(node in G.nodes() for node in tri):
                for j, node in enumerate(tri):
                    #print(tri)
                    if j == 0:
                        tri2 = (tri[1],tri[2])
                    elif j == 1:
                        tri2 = (tri[0],tri[2])
                    elif j == 2:
                        tri2 = (tri[0],tri[1])
                    
                    #print(tri2)
                    
                    if node in neighs:
                        neighs[node].append(tri2)
                    else:
                        neighs[node] = []
                        neighs[node].append(tri2)  
                
        print(neighs)
        
        return neighs


#To open a network which is already generated

def open_network(alpha):
    # 1) Lee triángulos
    triangles = [
        tuple(map(int, line.split()))
        for line in open(f"triangles{alpha}_ER.txt")
    ]
    # 2) Inicia un defaultdict
    neighs = defaultdict(list)
    # 3) Lee vecinos par a par
    for i, line in enumerate(open(f"pairwise_neighs{alpha}_ER.txt")):
        neighs[i] = list(map(int, line.split()))
    # 4) Añade HOIs sin preocuparte de claves inexistentes
    for a, b, c in triangles:
        neighs[a].append((b, c))
        neighs[b].append((a, c))
        neighs[c].append((a, b))
    return dict(neighs)

def evolution_mixed(states, neighs):
    #print(states)
    node = random.sample(list(states),1)
    node = node[0]
    #print(node)
    link_chosen = random.sample(neighs[node],1)[0]
    #print(node,link_chosen)
    
    if type(link_chosen) == int:
        #print(link_chosen)
        i = states[node]
        j = states[link_chosen]
        #print(i,j)
        #print(H[i,j])    
        u = np.random.uniform(0,1)
        #print(u)
        if u < H[i,j]:
            states[node]=i
        else:
            states[node]=j
            
    if type(link_chosen) == tuple:
        #print(link_chosen)
        #print(link_chosen[0],link_chosen[1])
        
        i = states[node]
        j = states[link_chosen[0]]
        k = states[link_chosen[1]]
        #print(i,j,k)

        B_i = 2*H[i,j]*H[i,k] + H[i,j]*H[j,k] + H[i,k]*H[k,j] #rate 1
        B_j = 2*H[j,i]*H[j,k] + H[j,i]*H[i,k] + H[j,k]*H[k,i] #rate 2
        B_k = 2*H[k,i]*H[k,j] + H[k,i]*H[i,j] + H[k,j]*H[j,i] #rate 3
        B_TOTAL = B_i + B_j + B_k
        B_i = B_i / B_TOTAL
        B_j = B_j / B_TOTAL
        B_k = B_k / B_TOTAL
        #print(B_i,B_j,B_k)
           
        u = np.random.uniform(0,1)
        #print(u)
        if u < B_i:
            states[node]=i
        elif u < B_i + B_j:
            #if j != i:
                #print(i,j)
                #print('now')
            states[node]=j
        else:
            #if k != i:
                #print(i,k)
                #print('now')
            states[node]=k
    
    return states


def simulation_mixed(times, times_eq, k, N, alphas, num_networks, num_runs_per_network):
    
    for alpha in alphas:
        
        for net_idx in range(num_networks):
            
            # 1. If we want to create a new network
            '''
            p1 = k/((N-1)*(1+alpha/(1-alpha)))
            p2 = (alpha/(1-alpha))*2/(N-2)*p1
            neighs = generate_network_mixed(N,p1,p2,alpha)
            '''
            # 2. If it is already created the network
            
            neighs = open_network(alpha)
            #print(neighs)
            
            
            
            for run in range(num_runs_per_network):
                
                # Initialize states for all nodes in G
                state = {node: np.random.randint(0, 3) for node in neighs}
                
                # Open a file to write density for each alpha, network, and run
                f0 = open(f"density_alpha_{alpha}_net_{net_idx}_run_size_"+str(N)+"_radius_"+str(k)+"_ER.txt", "w") 
                
                X0 = []
                X1 = []
                X2 = []
                TIMES = []
                
                # Run equilibrium steps
                for i in range(times_eq):
                    for j in range(len(state)):
                        state_new = evolution_mixed(state, neighs)  # INTERACTION TYPE
                        state = state_new
                
                # Main simulation loop
                for t in range(times):
                    TIMES.append(t)
                    for j in range(len(state)):
                        state_new = evolution_mixed(state, neighs)
                        state = state_new    
                    
                    # Count densities of each state
                    x0 = 0
                    x1 = 0
                    x2 = 0
                    for i in state:
                        if state[i] == 0:
                            x0 += 1
                        elif state[i] == 1:
                            x1 += 1
                        elif state[i] == 2:
                            x2 += 1                

                    X0.append(x0/len(state))
                    X1.append(x1/len(state))
                    X2.append(x2/len(state))

                    # Break if one of the states vanishes
                    if x0 < 1e-12 or x1 < 1e-12 or x2 < 1e-12:
                        break
                    
                    # Write the time and densities to file
                    f0.write(f"{t} {X0[-1]} {X1[-1]} {X2[-1]}\n")

                f0.close()  # Close the file for the current alpha, network, and run

    return

#To just generate the network

N = 1000
k = 10
alpha = 0.25
p1 = k/((N-1)*(1+alpha/(1-alpha)))
p2 = (alpha/(1-alpha))*2/(N-2)*p1

neighs = generate_network_mixed(N,p1,p2,alpha) 

#To run the simulation
'''
N = 1000
k = 10
alphas = [0.25]

times = 100
times_eq = 0

num_networks = 1
num_runs_per_network = 1

simulation_mixed(times,times_eq, k, N, alphas, num_networks, num_runs_per_network)
'''
'''
N = 10000
k = 20
alphas = [0.05,0.075,0.1,0.15]

times = 2400
times_eq = 100

num_networks = 1
num_runs_per_network = 20

simulation_mixed(times,times_eq, k, N, alphas, num_networks, num_runs_per_network)
'''
