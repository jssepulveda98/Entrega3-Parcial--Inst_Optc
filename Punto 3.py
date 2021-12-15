import matplotlib.pyplot as plt
import numpy as np
import time
import cv2

def Tmatrix(deltax0, deltay0, M,N):
    """
    Incident wave and transmittance function
    In this case: plane wave and circular aperture 
    """
    x=np.arange(-M,M)
    y=np.arange(-N,N)
    x,y=np.meshgrid(x,y)
    lim=750**2  #Radius of 750um
    A=701*deltax0
    B=445*deltay0
    T_matrix=(deltax0*x-A)**2 + (deltay0*y+B)**2
    T_matrix[np.where(T_matrix<=lim)]=1
    T_matrix[np.where(T_matrix>lim)]=0

    return T_matrix

def Ref(w_length, deltax0, deltay0, M,N):
    """
    Reference wave
    """
    x=np.arange(-M,M)
    y=np.arange(-N,N)
    x,y=np.meshgrid(x,y)

    k= 2*np.pi/w_length
    thetax=0.068284  #3.9124 degrees
    thetay=0.04336   #2.48449 degrees

    Ref=np.exp(1j*k*deltax0*x*np.sin(thetax))*np.exp(-1j*k*deltay0*y*np.sin(thetay))

    R=np.fft.fft2(Ref)
    R=np.fft.fftshift(R)

    return R
"""
U=incident wave
z=entrance plane to detector plane distance
w_length= wavelentgth
deltax=pixel size detector plane
deltay=pixel size detector plane
deltax0=pixel size entrance plane
deltay0=pixel size entrance plane
M*2=number of pixels in the x axis
N*2=number of pixels in the y axis
M*2xN*2=number of pixels entrance plane
"""

M=1024 #Number of pixels=M*2
N=1024 #Number of pixels=N*2
z=2500  #(2.5mm)
w_length=0.633   #All units in um
deltax0=3.18 
deltay0=3.18

deltax=(w_length*z)/(2*M*deltax0)
deltay=(w_length*z)/(2*M*deltay0)

print("deltax:",deltax)
print("deltay:",deltay)
tic=time.time()

T=Tmatrix(deltax0, deltay0, M,N)
holo= cv2.imread("Hologram.tiff",0)
FTholo= np.fft.fft2(holo)
FTholo= np.fft.fftshift(FTholo)
Filter=T*FTholo
InvFT=np.fft.ifft2(Filter)
R=Ref(w_length, deltax0, deltay0, M,N)

Iholo=np.log(np.abs(FTholo)**2)
IFilter=np.log((np.abs(Filter)**2)+0.0001)
IInvFT=np.abs(InvFT)**2
IR=np.log(np.abs(R)**2)

plt.imsave("Transmittance.png",T, cmap='gray') 
plt.imsave("Fourier Transform Hologram.png",Iholo, cmap='gray')   
plt.imsave("FT Hologram filtered.png",IFilter, cmap='gray')  
plt.imsave("Inverse FT filtered.png",IInvFT, cmap='gray')  
plt.imsave("FT Reference.png",IR, cmap='gray')   