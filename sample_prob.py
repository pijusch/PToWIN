from dionysus import *



simplices = []


with open("sample_prob.csv","r") as inp:
        for line in inp:
                x,y,z = map(float,line.split())
                simplices.append(Simplex((int(x),),0))
                simplices.append(Simplex((int(y),),0))
                simplices.append(Simplex((int(x),int(y)),z))

simplices.append(Simplex((1,8),1))
f1 = Filtration(simplices,data_dim_cmp)
p1 = StaticPersistence(f1)
p1.pair_simplices()

diagrams1 =  init_diagrams(p1,f1)

print diagrams1[0]


'''
Hello,

I'll be reaching Bonn around 15 December and I need the room till May 31st. Something about me: I am a 21 years old college student from India, I am coming to Bonn for a semester exchange (funded) program, I am a DAAD scholar, a happy go lucky person and a cleanliness freak. I don't speak German very well and a beginner in this language. (I do speak English very well though).

'''
