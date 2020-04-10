import math

for i in range(56):
    v = 0
    file = open('./sortedPosting/vector_{}.txt'.format(i),'r')
    data = file.read().split('\n')
    for d in data:
        if(d!=''):
            v+=pow(float(d),2)
    print("{}->{}".format(i,math.sqrt(v)))
