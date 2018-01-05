# code to find intersection of graphs

from __future__ import division
from dionysus import *
import numpy as np
import matplotlib.pyplot as plt

N = 172

distance_mat = np.empty([N,N])
distance_mat.fill(0)


#Function for calssical multidimension scaling
def cmdscale(D):
    """                                                                                       
    Classical multidimensional scaling (MDS)                                                  
                                                                                               
    Parameters                                                                                
    ----------                                                                                
    D : (n, n) array                                                                          
        Symmetric distance matrix.                                                            
                                                                                               
    Returns                                                                                   
    -------                                                                                   
    Y : (n, p) array                                                                          
        Configuration matrix. Each column represents a dimension. Only the                    
        p dimensions corresponding to positive eigenvalues of B are returned.                 
        Note that each dimension is only determined up to an overall sign,                    
        corresponding to a reflection.                                                        
                                                                                               
    e : (n,) array                                                                            
        Eigenvalues of B.                                                                     
                                                                                               
    """
    # Number of points                                                                        
    n = len(D)
 
    # Centering matrix                                                                        
    H = np.eye(n) - np.ones((n, n))/n
 
    # YY^T                                                                                    
    B = -H.dot(D**2).dot(H)/2
 
    # Diagonalize                                                                             
    evals, evecs = np.linalg.eigh(B)
 
    # Sort by eigenvalue in descending order                                                  
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
 
    # Compute the coordinates using positive-eigenvalued components only                      
    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)
 
    return Y, evals


for i in range(1,N+1):
        for j in range(i+1,N+1):
                input1 = "/home/mypc/Desktop/input_files/"+str(i)+".txt"        
                mat = np.empty([1000,1000])
                mat.fill(0)
                ct1 =0
                with open(input1,"r") as inp:
                        for line in inp:
                                ct1 = ct1 + 1
                                x,y,z, = map(float,line.split())
                                mat[int(x)][int(y)] = 1
                input2 = "/home/mypc/Desktop/input_files/"+str(j)+".txt"
                mat2 = np.empty([1000,1000])
                mat2.fill(0)
                ct2 = 0
                inter = 0
                with open(input2,"r") as inp:
                        for line in inp:
                                ct2 = ct2 + 1
                                x,y,z = map(float,line.split())
                                if mat[int(x)][int(y)] == 1:
                                        inter = inter + 1
                distance_mat[i-1][j-1] = ct1+ct2 -2*inter
                distance_mat[j-1][i-1] = ct1+ct2 -2*inter

#dimension reduction
a,b = cmdscale(distance_mat)
printar = np.empty([N])
printar.fill(0)

print b
#selecting the first dimension
for i in range(N):
        for j in range(0,5):
                printar[i] = printar[i] + a[i][j]*a[i][j]
        printar[i] = np.sqrt(printar[i])
        #print i+1,a[i][0]

fig = plt.figure()
ax = fig.add_subplot(111)

count = range(1,N+1)
plt.plot(count,printar,'-o')
i=1
for xy in zip(count,printar):                                       # <--
    ax.annotate(i, xy=xy, textcoords='data') # <--
    i+=1
plt.ylabel('reduced dimension')
plt.show()


