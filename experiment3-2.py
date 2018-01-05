        #Time Varying Graph Analysis using TDA


from __future__ import division
from dionysus import *
import numpy as np
import matplotlib.pyplot as plt


num = 172


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


#Function to calculate the persistence diagrams given the filename
def compute_matrix(_file_):
        ar = np.empty([1000])
        ar.fill(0)
        lis = []
        with open(_file_,"r") as inp:
                for line in inp:
                        x,y,z = map(float,line.split())
                        if ar[int(x)]==0:
                                lis.append(int(x))
                                ar[int(x)] = 1
                        if ar[int(y)]==0:
                                lis.append(int(y))
                                ar[int(y)] = 1
        ar.fill(0)
        for i in range(len(lis)):
                ar[lis[i]] = i
        
        mat = np.empty([len(lis),len(lis)])
        mat.fill(10)
        with open(_file_,"r") as inp:
                for line in inp:
                        x,y,z = map(float,line.split())
                        mat[int(ar[int(x)])][int(ar[int(y)])] = z
        
        for k in range(len(lis)):
                for i in range(len(lis)):
                        for j in range(len(lis)):
                                if mat[i][k] + mat[k][j] < mat[i][j]:
                                        mat[i][j] = mat[i][k] + mat[k][j]

        simplices = []

        for i in range(len(mat)):
                simplices.append(Simplex((i,),0))
        for i in range(len(mat)):
                for j in range(i+1,len(mat)):
                        simplices.append(Simplex((i,j),mat[i][j]))

        f = Filtration(simplices,data_dim_cmp)
        p = StaticPersistence(f)
        p.pair_simplices()
        diagrams =  init_diagrams(p,f)

        return diagrams


#matrix where wasserstein distances are stored
mat  = np.empty([num,num])
mat.fill(0)


#Calculating wasserstein distance
for i in range(num):

        input1 = "/home/mypc/Desktop/tvg_files/input_files2/"+str(i+1)+".txt"

        diagrams1 =  compute_matrix(input1)
        j = i+1
        for j in range(i+1,num):
                        input2 = "/home/mypc/Desktop/tvg_files/input_files2/"+str(j+1)+".txt"
                        diagrams2 =  compute_matrix(input2)
                        dim = 0
                        p = 1
                        mat[i][j] = wasserstein_distance(diagrams1[dim],diagrams2[dim],p)
                        #mat[i][j] = bottleneck_distance(diagrams1[dim],diagrams2[dim])
                        mat[j][i] = mat[i][j]
                        #print i+1,j+1,wasserstein_distance(diagrams1[dim],diagrams2[dim],p)


#dimension reduction
a,b = cmdscale(mat)
printar = np.empty([num])

#selecting the first dimension
for i in range(num):
        printar[i] = a[i][0]
        print i+1,a[i][0]

fig = plt.figure()
ax = fig.add_subplot(111)



#plotting the 1st colum after applying cmd scaling
count = range(1,num+1)
plt.plot(count,printar,'-o')
i=1
for xy in zip(count,printar):                                       # <--
    ax.annotate(i, xy=xy, textcoords='data') # <--
    i+=1
plt.ylabel('reduced dimension')
plt.show()
