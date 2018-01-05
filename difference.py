# code to find intersection of graphs

from dionysus import *
import numpy as np

s = 111
e = 140 

input1 = "/home/mypc/Desktop/input_files/"+str(s)+".txt"

mat = np.empty([1000,1000])
mat.fill(0)

with open(input1,"r") as inp:
        for line in inp:
                x,y,z = map(float,line.split())
                mat[int(x)][int(y)] = 1
        
for i in range(s+1,e+1):
        input2 = "/home/mypc/Desktop/input_files/"+str(i)+".txt"
        mat2 = np.empty([1000,1000])
        mat2.fill(0)
        with open(input2,"r") as inp:
                for line in inp:
                         x,y,z = map(float,line.split())
                         if mat[int(x)][int(y)] != 0:
                                mat2[int(x)][int(y)] = 1
        mat = mat2



ar = np.empty([e-s+1])
for i in range(s,e+1):
        input1 = "/home/mypc/Desktop/input_files/"+str(i)+".txt"
        ct =0
        with open(input1,"r") as inp:
                for line in inp:
                        x,y,z = map(float,line.split())
                        if mat[int(x)][int(y)] == 0:
                                ct = ct + 1
        ar[i-s] = ct

ct = 0
for i in range(100):
        for j in range(i,100):
                if mat[i][j] == 1:
                        ct = ct + 1

print ct
print ar
print np.var(ar)
                        
