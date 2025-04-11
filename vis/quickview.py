import ptrReader
import numpy as np 
import sys,os
import matplotlib.pyplot as plt
import struct
from tqdm import tqdm
plt.style.use('dark_background')

re=6378137.0
files=sys.argv[2::]
for file in tqdm(files):
    x,y,z,vx,vy,vz=ptrReader.read_ptr2_file(file)
    x/=re
    y/=re
    z/=re
    #XY Plot
    if (sys.argv[1]=="XY"):
        plt.scatter(x,y,s=0.1,c='w')
    elif (sys.argv[1]=="XZ"):
        plt.scatter(x,z,s=0.1,c='w')
lim=15
if (sys.argv[1]=="XY"):
    # plt.title(f"Equatorial Slice (GC) | Time = {time:.2f} s")
    plt.xlabel("X [RE]")
    plt.ylabel("Y [RE]")
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    circle1 = plt.Circle((0, 0), 1, color='b',alpha=0.5)
    plt.gca().add_patch(circle1)
    plt.savefig("plot_xy.png")
elif (sys.argv[1]=="XZ"):
    # plt.title(f"Meridional Slice (GC) | Time = {time:.2f} s")
    plt.xlabel("X [RE]")
    plt.ylabel("Z [RE]")
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    circle1 = plt.Circle((0, 0), 1, color='b',alpha=0.5)
    plt.gca().add_patch(circle1)
    plt.savefig("plot_xz.png")
