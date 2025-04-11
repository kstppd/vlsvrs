import numpy as np 
import struct
import sys,os


def read_ptr2_file(file):
    f=open(file, "rb")
    _size=f.read(8)
    _datasize=f.read(8)
    size=int.from_bytes(_size, "little")
    datasize=int.from_bytes(_datasize, "little")
    datatype=np.float32;
    if (datasize==8):
        datatype=np.float64;
    x=np.fromfile(f,count=size,dtype=datatype)
    y=np.fromfile(f,count=size,dtype=datatype)
    z=np.fromfile(f,count=size,dtype=datatype)
    vx=np.fromfile(f,count=size,dtype=datatype)
    vy=np.fromfile(f,count=size,dtype=datatype)
    vz=np.fromfile(f,count=size,dtype=datatype)
    return x,y,z,vx,vy,vz
