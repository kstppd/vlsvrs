import ptrReader
import numpy as np 
import sys,os
import matplotlib.pyplot as plt
from multiprocessing import Pool
plt.style.use('dark_background')


def plotFile(input):
    file,cnt=input
    x,y,z,vx,vy,vz=ptrReader.read_ptr2_file(file)
    print(x)
    x/=re
    y/=re
    z/=re

    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16,9))
    #XY Plot
    axes[0].scatter(x,y,s=0.1,c='w')
    axes[0].set_title("Equatorial Slice ")
    axes[0].set_xlabel("X [RE]")
    axes[0].set_ylabel("Y [RE]")
    axes[1].scatter(x,z,s=0.1,c='w')
    axes[1].set_title("Meridional Slice")
    axes[1].set_xlabel("X [RE]")
    axes[1].set_ylabel("Z [RE]")
    lim=45
    axes[0].set_xlim(-lim,lim)
    axes[0].set_ylim(-lim,lim)
    axes[1].set_xlim(-lim,lim)
    axes[1].set_ylim(-lim,lim)
    axes[0].set_aspect(1)
    axes[1].set_aspect(1)
    circle = plt.Circle((0, 0), 1, color='b',alpha=0.5)
    circle1 = plt.Circle((0, 0), 1, color='b',alpha=0.5)
    axes[0].add_patch(circle)
    axes[1].add_patch(circle1)
    plt.savefig("pop_"+str(cnt).zfill(7)+".png",dpi=500)
    plt.clf()
    plt.close()


re=6378137.0
files=sys.argv[1::]
pool=Pool(8)
index=np.arange(0,len(files))
pool.map(plotFile,zip(files,index))

