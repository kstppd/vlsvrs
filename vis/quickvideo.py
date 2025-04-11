import numpy as np 
import sys,os
import matplotlib.pyplot as plt
import struct
from multiprocessing import Pool
plt.style.use('dark_background')
import ptrReader

def plotFile(input):
    file,cnt=input
    x,y,z,vx,vy,vz=ptrReader.read_ptr2_file(file)
    print(x)
    x/=re
    y/=re
    z/=re

    #XY Plot
    if (sys.argv[1]=="XY"):
        plt.scatter(x,y,s=0.1,c='w')
        plt.title("Equatorial Slice | "+"Time= "+str(time)+" s")
        plt.xlabel("X [RE]")
        plt.ylabel("Y [RE]")
    elif (sys.argv[1]=="XZ"):
        plt.scatter(x,z,s=0.1,c='w')
        plt.title("Meridional Slice | "+"Time= "+str(time)+" s")
        plt.xlabel("X [RE]")
        plt.ylabel("Z [RE]")
    lim=45
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    circle1 = plt.Circle((0, 0), 1, color='b',alpha=0.5)
    plt.gca().add_patch(circle1)
    plt.savefig("pop_"+str(cnt).zfill(7)+".png",dpi=300)
    plt.clf()



re=6378137.0
files=sys.argv[2::]
pool=Pool()
index=np.arange(0,len(files))
pool.map(plotFile,zip(files,index))

