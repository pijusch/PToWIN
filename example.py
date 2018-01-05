from dionysus import *



simplices = []


with open("input1","r") as inp:
        for line in inp:
                x,y,z = map(float,line.split())
                simplices.append(Simplex((int(x),),0))
                simplices.append(Simplex((int(y),),0))
                simplices.append(Simplex((int(x),int(y)),z))

'''
simplices.append(Simplex((1,),.04))
simplices.append(Simplex((2,),.01))
simplices.append(Simplex((3,),.05))
simplices.append(Simplex((4,),.02))
'''

f1 = Filtration(simplices,data_dim_cmp)
p1 = StaticPersistence(f1)
p1.pair_simplices()
'''
smap = p1.make_simplex_map(f1)

for i in p1:
    if i.sign():
        birth = smap[i]
        if i.unpaired():
            print birth.dimension(), birth.data, "inf"
            continue

        death = smap[i.pair()]
        print birth.dimension(), birth.data, death.data
'''        
diagrams1 =  init_diagrams(p1,f1)


simplices = []



with open("input2","r") as inp:
#with open("input","r") as inp:
#        for i in range(no_of_nodes):
#                simplices.append(Simplex((i,),0))
        for line in inp:
                x,y,z = map(float,line.split())           
                simplices.append(Simplex((int(x),),0))
                simplices.append(Simplex((int(y),),0))  
                simplices.append(Simplex((int(x),int(y)),z))
'''
simplices.append(Simplex((1,),.02))
simplices.append(Simplex((2,),.01))
simplices.append(Simplex((3,),.02))
simplices.append(Simplex((4,),.03))
'''

#simplices.append(Simplex((0,1,2),1))          #Square filled H1 becomes zero at 10
#simplices.append(Simplex((0,2,3),1))

f2 = Filtration(simplices,data_dim_cmp)
p2 = StaticPersistence(f2)
p2.pair_simplices()
'''
smap = p2.make_simplex_map(f2)


for i in p2:
    if i.sign():
        birth = smap[i]
        if i.unpaired():
            print birth.dimension(), birth.data, "inf"
            continue

        death = smap[i.pair()]
        print birth.dimension(), birth.data, death.data

'''
diagrams2 =  init_diagrams(p2,f2)
print diagrams1[0]
print "\n"
print diagrams2[0]


print bottleneck_distance(diagrams1[0],diagrams2[0])
