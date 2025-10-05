import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.colors

#Define map colors
levels = [0, 1, 2, 3]
colors = ['b', 'm', 'y']

norm = matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=len(colors), clip=False)
cmap = matplotlib.colors.ListedColormap(colors)

g = 3 #number species
H = np.matrix([[0.5, 0.34, 0.76], [0.66,0.5,0.25], [0.24,0.75,0.5]]) #interaction matrix

def Square_Lattice(L):
    return np.random.choice([0, 1, 2], size=(L, L))

def evolution_mixed(state, L, alpha, radius): 
    x = random.randint(0, L-1)
    y = random.randint(0, L-1)
    #print(x,y)
    #print(state[x,y])
    
    v = np.random.uniform(0,1)
    #print('alpha',v)
    if v < alpha:
        
        #RANGE INTERACTION
        
        #1
        if not radius == L:
        
            i = state[x,y]    
            select_1x = np.random.randint(-radius,radius + 1)
            select_1y = np.random.randint(-radius,radius + 1)
            while select_1x == 0 and select_1y == 0: #cannot be itself
                select_1x = np.random.randint(-radius,radius + 1)
                select_1y = np.random.randint(-radius,radius + 1)
            j = state[(x+select_1x)%L,(y+select_1y)%L]
            
            select_2x = np.random.randint(-radius,radius + 1)
            select_2y = np.random.randint(-radius,radius + 1)
            while select_2x == 0 and select_2y == 0 or select_2x == select_1x and select_2y == select_1y: #cannot be itself neither the same as the other neighbour chosen
                select_2x = np.random.randint(-radius,radius + 1)
                select_2y = np.random.randint(-radius,radius + 1)
            k = state[(x+select_2x)%L,(y+select_2y)%L]
            
        
        #2
        elif radius == L:
        
            i = state[x,y]
            j = state[random.randint(0, L-1),random.randint(0, L-1)]
            k = state[random.randint(0, L-1),random.randint(0, L-1)]
        
        
        #DYNAMICS 
        
        #print(i,j,k)
        B_i = 2*H[i,j]*H[i,k] + H[i,j]*H[j,k] + H[i,k]*H[k,j]
        B_j = 2*H[j,i]*H[j,k] + H[j,i]*H[i,k] + H[j,k]*H[k,i]
        B_k = 2*H[k,i]*H[k,j] + H[k,i]*H[i,j] + H[k,j]*H[j,i]
        B_TOTAL = B_i + B_j + B_k
        B_i = B_i / B_TOTAL
        B_j = B_j / B_TOTAL
        B_k = B_k / B_TOTAL
        #print(B_i,B_j,B_k)
        u = np.random.uniform(0,1)
        #print(u)
        if u < B_i:
            state[x,y]=i
        elif u < B_i + B_j:
            state[x,y]=j
        else:
            state[x,y]=k
    else:
        
        #RANGE INTERACTION
        
        #1
        if not radius == L:
            
            i = state[x,y]
            select_1x = np.random.randint(-radius,radius + 1)
            select_1y = np.random.randint(-radius,radius + 1)
            while select_1x == 0 and select_1y == 0: #cannot be itself
                select_1x = np.random.randint(-radius,radius + 1)
                select_1y = np.random.randint(-radius,radius + 1)
            #print(select_1x,select_1y)
            j = state[(x+select_1x)%L,(y+select_1y)%L]
            #print(j)
        
        #3
        elif radius == L:
            
            i = state[x,y]
            j = state[random.randint(0, L-1),random.randint(0, L-1)]
            #print(i,j)
        
        
        # DYNAMICS
        u = np.random.uniform(0,1)
        #print(u)
        if u < H[i,j]:
            state[x,y]=i
        else:
            state[x,y]=j

    #print(state)
    return state
'''
L = 10
state = Square_Lattice(L)
print(state)
evolution_triplewise(state, L)
print(state)
'''

def simulation(L,times,times_eq, alphas, num_networks, num_runs_per_network, radius):    

    for alpha in alphas:
        
        for net_idx in range(num_networks):
            
            for run in range(num_runs_per_network):
            
                state = Square_Lattice(L)
                #print(state)
                #plt.imshow(state, cmap=cmap)
                #plt.show()
                 
                f0 = open(f"density_alpha_{alpha}_net_{net_idx}_run_{run}_size_"+str(L)+"_radius_"+str(radius)+"_Lattice.txt", "w")
                X0 = []
                X1 = []
                X2 = []
                TIMES = []
                for _ in range(times_eq):
                    for _ in range(L**2):
                        state_new = evolution_mixed(state, L, alpha, radius) 
                        state = state_new
                for i in range(times):
                    TIMES.append(i)
                    for _ in range(L**2):
                        state_new = evolution_mixed(state, L, alpha, radius)
                        state = state_new
                    x0 = 0
                    x1 = 0
                    x2 = 0
                    for i in range(L):
                        for j in range(L):
                            if state[i,j] == 0:
                                x0 += 1
                            if state[i,j] == 1:
                                x1 += 1
                            if state[i,j] == 2:
                                x2 += 1
                    X0.append(x0/L**2)
                    X1.append(x1/L**2)
                    X2.append(x2/L**2)
                    
                for j in range(times):
                    f0.write(str(j) + ' ' + str(X0[j]) + ' ' + str(X1[j]) + ' ' + str(X2[j]) + "\n")
                f0.close()   

            plt.pcolormesh(state, cmap=cmap, norm=norm)
            plt.draw()
            plt.savefig('lattice_simulation_'+str(radius)+'.svg', dpi=600)
            plt.show()

    return

L=100
radius = L
times = 1000
times_eq = 100
alphas = [0.1]

num_networks = 1
num_runs_per_network = 1

simulation(L,times,times_eq,alphas, num_networks, num_runs_per_network, radius)










