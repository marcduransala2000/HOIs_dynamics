import matplotlib.pyplot as plt
import numpy as np
import collections
import sys
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib as mpl
import math
from scipy.stats import gaussian_kde
from scipy.spatial import Delaunay
from sympy import Point, Polygon
from sklearn.decomposition import PCA

import os
os.chdir(r'C:\Users\mduran\Desktop\MSc\TFM\renewed_model')

L= 100
N = 10000
network = 'Lattice'
radius = L
k = 20

def load_data(alpha, net_idx, run, network, radius):
    # Generate the file name based on the alpha, network index, and run number
    if network == 'ER':
        file_name = f"density_alpha_{alpha}_net_{net_idx}_run_size_"+str(N)+"_radius_"+str(k)+"_ER.txt"
    elif network == 'Lattice':
        file_name = f"density_alpha_{alpha}_net_{net_idx}_run_{run}_size_"+str(L)+"_radius_"+str(radius)+"_Lattice.txt"
    
    # Load the data from the file
    data = np.loadtxt(file_name)
    
    return data
    
alphas = [0,0.1]  # Specify the alpha value
net_idxs = [0]  # Specify the network index (e.g., 0 for the first network)
runs = range(0,1) # Specify the run number (e.g., 0 for the first run)
time = 1000

for alpha in alphas:
    AREAS = []
    for net_idx in net_idxs:
        for run in runs:

            # Load the corresponding density data
            data = load_data(alpha, net_idx, run, network, radius)            
            
            # Dynamics plot
            plt.figure()  
            TIMES = data[0:time, 0]
            X0 = data[0:time, 1]
            X1 = data[0:time, 2]
            X2 = data[0:time, 3]
            
            plt.plot(TIMES,X0,'b')
            plt.plot(TIMES,X1,'m')
            plt.plot(TIMES,X2,'y')  
            plt.ylim(0,1)
            plt.xlim(0,time)
            plt.xlabel(r"$t$", fontsize = '16')
            plt.ylabel(r"$x_i(t)$", fontsize = '16')
            #plt.savefig('short_range_pairwise.png', dpi=400) 
            plt.show()
            
            expected_time_steps = time # Check if they survived
            if data.shape[0] < expected_time_steps:
                print(f"Run {run}: Data has only {data.shape[0]} time steps")
                continue  # Skip this run
            
            # 3D plot
            plt.figure()  
            TIMES = data[0:time, 0]
            X0 = data[0:time, 1]
            X1 = data[0:time, 2]
            X2 = data[0:time, 3]
            
            ax = plt.axes(projection='3d')
            ax.plot3D(X0, X1, X2, '.')
            
            ax.set_xlim(0,0.8)
            ax.set_ylim(0,0.8)
            ax.set_zlim(0,0.8)
            
            ax.set_xlabel(r"$x_1$", fontsize = 14)
            ax.set_ylabel(r"$x_2$", fontsize = 14)
            ax.set_zlabel(r"$x_3$", fontsize = 14)
            
            #plt.savefig('simplex3D_longrange.png', dpi=300)
            plt.show()
            
            
            # 2D plot
            Y1 = []
            Y2 = []
            for i in range(len(X0)):
                Y1.append(np.sqrt(1/2)*(X0[i]*(-1)+X1[i]*1))
                Y2.append(np.sqrt(2/3)*(X0[i]*(-1/2)+X1[i]*(-1/2)+X2[i]))
            
            
             
            plt.plot(Y1, Y2,'.')
            plt.xlim(-0.5,0.5)
            plt.ylim(-0.5,0.5)
            plt.ylabel(r'$Y_2$', fontsize = 11.5)
            plt.xlabel(r'$Y_1$', fontsize = 11.5)
            
            plt.show()
            
            #AREA            
            
            def alpha_shape(points, alpha, only_outer=True):
                """
                Compute the alpha shape (concave hull) of a set of points.
                :param points: np.array of shape (n,2) points.
                :param alpha: alpha value.
                :param only_outer: boolean value to specify if we keep only the outer border
                or also inner edges.
                :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
                the indices in the points array.
                """
                assert points.shape[0] > 3, "Need at least four points"
            
                def add_edge(edges, i, j):
                    """
                    Add an edge between the i-th and j-th points,
                    if not in the list already
                    """
                    if (i, j) in edges or (j, i) in edges:
                        # already added
                        assert (j, i) in edges, "Can't go twice over same directed edge"
                        if only_outer:
                            # if both neighboring triangles are in shape, it's not a boundary edge
                            edges.remove((j, i))
                        return
                    edges.add((i, j))
            
                tri = Delaunay(points)
                edges = set()
                # Loop over triangles:
                # ia, ib, ic = indices of corner points of the triangle
                for ia, ib, ic in tri.simplices:
                    pa = points[ia]
                    pb = points[ib]
                    pc = points[ic]
                    # Computing radius of triangle circumcircle
                    # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
                    a = np.linalg.norm(pa - pb)
                    b = np.linalg.norm(pb - pc)
                    c = np.linalg.norm(pc - pa)
                    s = (a + b + c) / 2.0
                    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                    circum_r = a * b * c / (4.0 * area)
                    if circum_r < alpha:
                        add_edge(edges, ia, ib)
                        add_edge(edges, ib, ic)
                        add_edge(edges, ic, ia)
                return edges
            
            def find_edges_with(i, edge_set):
                i_first = [j for (x,j) in edge_set if x==i]
                i_second = [j for (j,x) in edge_set if x==i]
                return i_first,i_second
            
            def stitch_boundaries(edges): #In order to organize all the points in the correct order
                edge_set = edges.copy()
                boundary_lst = []
                while len(edge_set) > 0:
                    boundary = []
                    edge0 = edge_set.pop()
                    boundary.append(edge0)
                    last_edge = edge0
                    while len(edge_set) > 0:
                        i,j = last_edge
                        j_first, j_second = find_edges_with(j, edge_set)
                        if j_first:
                            edge_set.remove((j, j_first[0]))
                            edge_with_j = (j, j_first[0])
                            boundary.append(edge_with_j)
                            last_edge = edge_with_j
                        elif j_second:
                            edge_set.remove((j_second[0], j))
                            edge_with_j = (j, j_second[0])  # flip edge rep
                            boundary.append(edge_with_j)
                            last_edge = edge_with_j
            
                        if edge0[0] == last_edge[1]:
                            break
            
                    boundary_lst.append(boundary)
                return boundary_lst
            
            def remove_from_array(base_array, test_array):
                for index in range(len(base_array)):
                    if np.array_equal(base_array[index], test_array):
                        base_array.remove(index)
                        break
                raise ValueError('remove_from_array(array, x): x not in array')
                
            # Constructing the input point data
            points = np.vstack([Y1, Y2]).T
            
            #REMOVING OUTLIERS
            outliers = 1 # number of outliers' layers removed
            
            for i in range(0,outliers):
            
                edges = alpha_shape(points, alpha=0.75, only_outer=True)
                edges = stitch_boundaries(edges)[0]
                
                points2 = points.tolist()
                points = points.tolist()
                for i,j in edges:
                    points.remove([points2[[i,j][0]][0],points2[[i,j][0]][1]])
                    
                points = np.array(points)
                
            
            edges = alpha_shape(points, alpha=0.75, only_outer=True)
            edges = stitch_boundaries(edges)[0]
            
            # Plotting the output
            plt.plot(points[:, 0], points[:, 1], '.')
            plt.ylabel(r'$Y_2$', fontsize = 13)
            plt.xlabel(r'$Y_1$', fontsize = 13)
            plt.xlim(-0.5,0.5)
            plt.ylim(-0.5,0.5)
            
            for i, j in edges:
                plt.plot(points[[i, j], 0], points[[i, j], 1], 'orange')  
                
            #plt.savefig('simplex2D.pdf')    
            plt.show()
            
            EDGES = []
            
            for i,j in edges:
                EDGES.append(points[[i, j]][0])
                EDGES.append(points[[i, j]][1])
            print(len(EDGES))  
            
            sympy_points = [Point(p[0], p[1]) for p in EDGES]
            
            # Create a Polygon from the points in EDGES
            polygon = Polygon(*sympy_points)
            
            # Calculate the area of the polygon
            area = polygon.area
            
            AREAS.append(round(area, 3))
            
            print(f"Area of the polygon: {round(area, 3)}")

 