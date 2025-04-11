import ptrReader
import numpy as np 
import sys,os
import matplotlib.pyplot as plt
import struct
from tqdm import tqdm
plt.style.use('dark_background')

re=6378137.0
files=sys.argv[1::]

x1,y1,z1,vx1,vy1,vz1=ptrReader.read_ptr2_file(files[0])
x2,y2,z2,vx2,vy2,vz2=ptrReader.read_ptr2_file(files[1])
x3,y3,z3,vx3,vy3,vz3=ptrReader.read_ptr2_file(files[2])

en1=np.sqrt(vx1*vx1+vy1*vy1+vz1*vz1)
en2=np.sqrt(vx2*vx2+vy2*vy2+vz2*vz2)
en3=np.sqrt(vx3*vx3+vy3*vy3+vz3*vz3)

fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
plt.suptitle("Energy Histogram Normalized")
axs[0].hist(en1,bins=50,label="low acc")
axs[1].hist(en2,bins=50,label="med acc")
axs[2].hist(en3,bins=50,label="high acc")
axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.savefig("final_state.png")


