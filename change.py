l = []

with open("edgelist_15017.txt","r") as inp:
        for line in inp:
                x,y,z = map(float,line.split())
                l.append(((int(x),int(y)),z))

d = dict(l)

ll = []

with open("edgelist_3002.txt","r") as inp:
        for line in inp:
                x,y,z =map(float,line.split())
                if d.has_key((int(x),int(y))):
                        ll.append(((int(x),int(y)),d[(int(x),int(y))]))
                        print "same"
                else:   
                        print "not same"
                        ll.append(((int(x),int(y)),z))

print l
print ll

