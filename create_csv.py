
with open("/home/mypc/Desktop/input_files/19.txt","r") as inp:
        for line in inp:
                x,y,z = map(float,line.split())
                print str(int(x))+','+str(int(y))
                

