import numpy as np


input1 = "/home/mypc/Dionysus/build/bindings/python/tvgs_new/11.txt"
input2 = "/home/mypc/Dionysus/build/bindings/python/tvgs_new/10.txt"

mat = np.empty([1000,1000])
mat.fill(0)

with open(input1,"r") as inp:
        for line in inp:
                x,y,z = map(float,line.split())
                mat[int(x)][int(y)] = 1

with open(input2,"r") as inp:
        for line in inp:
                x,y,z = map(float,line.split())
                if mat[int(x)][int(y)] == 0:
                        print int(x),int(y)
                

