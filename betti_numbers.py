from dionysus import *



simplices = []

N = 172

for i in range(1,N+1):
        input_x = "/home/mypc/Desktop/input_files/"+str(i)+".txt"

        with open(input_x,"r") as inp:
                for line in inp:
                        x,y,z = map(float,line.split())
                        simplices.append(Simplex((int(x),),0))
                        simplices.append(Simplex((int(y),),0))
                        simplices.append(Simplex((int(x),int(y)),z))



        f1 = Filtration(simplices,data_dim_cmp)
        p1 = StaticPersistence(f1)
        p1.pair_simplices()
        smap = p1.make_simplex_map(f1)
        c = 0
        for j in p1:
            if j.sign():
                birth = smap[j]
                if j.unpaired() and birth.dimension()==0:
                    c = c + 1
                    continue
        print i,c

