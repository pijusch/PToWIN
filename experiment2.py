#Time Varying Graph Analysis using TDA


from __future__ import division
from dionysus import *
import numpy as np
import matplotlib.pyplot as plt

num = 50


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
                        mat[ar[int(x)]][ar[int(y)]] = z
        
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

        input1 = "/home/mypc/Desktop/input_files/"+str(i+1)+".txt"

        diagrams1 =  compute_matrix(input1)
        j = i+1
        for j in range(i+1,num):
                        input2 = "/home/mypc/Desktop/input_files/"+str(j+1)+".txt"
                        diagrams2 =  compute_matrix(input2)
                        dim = 0
                        p = 1
                        mat[i][j] = wasserstein_distance(diagrams1[dim],diagrams2[dim],p)
                        mat[j][i] = mat[i][j]
                        #print i+1,j+1,wasserstein_distance(diagrams1[dim],diagrams2[dim],p)


#dimension reduction
a,b = cmdscale(mat)
printar = np.empty([num])

#selecting the first dimension
for i in range(num):
        printar[i] = a[i][0]

#plotting the 1st colum after applying cmd scaling
plt.plot(printar,'-o')
plt.ylabel('reduced dimension')
plt.show()



        






#############################################################################################################################################

#rough code

'''
simplices = []

input1 = "/home/mypc/Dionysus/build/bindings/python/tvgs_new/1.txt"
input2 = "/home/mypc/Dionysus/build/bindings/python/tvgs_new/2.txt"

check = np.empty([1000])
check.fill(0)

with open(input1,"r") as inp:
        for line in inp:
                x,y,z = map(float,line.split())
                simplices.append(Simplex((int(x),),0))
                simplices.append(Simplex((int(y),),0))
                simplices.append(Simplex((int(x),int(y)),z))

f1 = Filtration(simplices,data_dim_cmp)
p1 = StaticPersistence(f1)
p1.pair_simplices()
diagrams1 =  init_diagrams(p1,f1)


simplices = []



with open(input2,"r") as inp:
#with open("input","r") as inp:
#        for i in range(no_of_nodes):
#                simplices.append(Simplex((i,),0))
        for line in inp:
                x,y,z = map(float,line.split())
                simplices.append(Simplex((int(x),),0))
                simplices.append(Simplex((int(y),),0))
                simplices.append(Simplex((int(x),int(y)),z))
                


#simplices.append(Simplex((0,1,2),1))          #Square filled H1 becomes zero at 10
#simplices.append(Simplex((0,2,3),1))

f2 = Filtration(simplices,data_dim_cmp)
p2 = StaticPersistence(f2)
p2.pair_simplices()
diagrams2 =  init_diagrams(p2,f2)


#print wasserstein_distance(diagrams1[0],diagrams2[0],1)
print bottleneck_distance(diagrams1[0],diagrams2[0])
print "\n"
print diagrams1[0]
print "\n"
print diagrams2[0]


#print "graph i","graph i+1","bottleneck","wasserstein"
'''

'''
for i in range(14):
        simplices = []

        input1 = "/home/mypc/Dionysus/build/bindings/python/tvgs_new/"+str(i+1)+".txt"
        input2 = "/home/mypc/Dionysus/build/bindings/python/tvgs_new/"+str(i+2)+".txt"

        with open(input1,"r") as inp:
                for line in inp:
                        x,y,z = map(float,line.split())
                        simplices.append(Simplex((int(x),),0))
                        simplices.append(Simplex((int(y),),0))
                        simplices.append(Simplex((int(x),int(y)),z))

        f1 = Filtration(simplices,data_dim_cmp)
        p1 = StaticPersistence(f1)
        p1.pair_simplices()
        diagrams1 =  init_diagrams(p1,f1)

        simplices = []



        with open(input2,"r") as inp:
        #with open("input","r") as inp:
        #        for i in range(no_of_nodes):
        #                simplices.append(Simplex((i,),0))
                for line in inp:
                        x,y,z = map(float,line.split())           
                        simplices.append(Simplex((int(x),),0))
                        simplices.append(Simplex((int(y),),0))  
                        simplices.append(Simplex((int(x),int(y)),z))

        #simplices.append(Simplex((0,1,2),1))          #Square filled H1 becomes zero at 10
        #simplices.append(Simplex((0,2,3),1))

        f2 = Filtration(simplices,data_dim_cmp)
        p2 = StaticPersistence(f2)
        p2.pair_simplices()
        diagrams2 =  init_diagrams(p2,f2)

        dim = 0
        p = 1
        print i+1,i+2,bottleneck_distance(diagrams1[dim],diagrams2[dim]),wasserstein_distance(diagrams1[dim],diagrams2[dim],p)
        #print i+1,i+2,wasserstein_distance(diagrams1[0],diagrams2[0],1)



'''



'''
mat  = np.empty([15,15])

for i in range(15):
        simplices = []

        input1 = "/home/mypc/Dionysus/build/bindings/python/tvgs_new/"+str(i+1)+".txt"

        with open(input1,"r") as inp:
                for line in inp:
                        x,y,z = map(float,line.split())
                        simplices.append(Simplex((int(x),),0))
                        simplices.append(Simplex((int(y),),0))
                        simplices.append(Simplex((int(x),int(y)),z))

        f1 = Filtration(simplices,data_dim_cmp)
        p1 = StaticPersistence(f1)
        p1.pair_simplices()
        diagrams1 =  init_diagrams(p1,f1)
        j = i+1
        for j in range(i+1,15):
                        simplices = []

                        input2 = "/home/mypc/Dionysus/build/bindings/python/tvgs_new/"+str(j+1)+".txt"

                        with open(input2,"r") as inp:
                        #with open("input","r") as inp:
                        #        for i in range(no_of_nodes):
                        #                simplices.append(Simplex((i,),0))
                                for line in inp:
                                        x,y,z = map(float,line.split())           
                                        simplices.append(Simplex((int(x),),0))
                                        simplices.append(Simplex((int(y),),0))  
                                        simplices.append(Simplex((int(x),int(y)),z))

                        #simplices.append(Simplex((0,1,2),1))          #Square filled H1 becomes zero at 10
                        #simplices.append(Simplex((0,2,3),1))

                        f2 = Filtration(simplices,data_dim_cmp)
                        p2 = StaticPersistence(f2)
                        p2.pair_simplices()
                        diagrams2 =  init_diagrams(p2,f2)

                        dim = 0
                        p = 1
                        mat[i][j] = bottleneck_distance(diagrams1[dim],diagrams2[dim])
                        print i+1,j+1,bottleneck_distance(diagrams1[dim],diagrams2[dim]),wasserstein_distance(diagrams1[dim],diagrams2[dim],p)
                        #print i+1,i+2,wasserstein_distance(diagrams1[0],diagrams2[0],1)


'''

'''
for i in range(15):
        for j in range(15):
                print mat[i][j],
                print "\t",
        print "\n"      
'''


'''

simplices = []

input1 = "/home/mypc/Dionysus/build/bindings/python/tvgs_new/1.txt"

diag1  = compute_matrix(input1)

print "\n"

simplices = []

input2 = "/home/mypc/Dionysus/build/bindings/python/tvgs_new/2.txt"

diag2  = compute_matrix(input2)

print wasserstein_distance(diag1[0],diag2[0],1)
'''


'''
check = np.empty([1000])
check.fill(0)

with open(input1,"r") as inp:
        for line in inp:
                x,y,z = map(float,line.split())
                simplices.append(Simplex((int(x),),0))
                simplices.append(Simplex((int(y),),0))
                simplices.append(Simplex((int(x),int(y)),z))

f1 = Filtration(simplices,data_dim_cmp)
p1 = StaticPersistence(f1)
p1.pair_simplices()
diagrams1 =  init_diagrams(p1,f1)
'''
