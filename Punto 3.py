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

    Ref=np.exp(-1j*k*deltax0*x*np.sin(thetax))*np.exp(1j*k*deltay0*y*np.sin(thetay))

    FTRef=np.fft.fft2(Ref)
    FTRef=np.fft.fftshift(FTRef)

    return Ref,FTRef

def Fresnel(Field, z, deltax0, deltay0, w_length, deltax, deltay, M, N):
    """
    Fresnel function
    """
    
    k= 2*np.pi/w_length
    
    x =np.arange(-M,M)
    y =np.arange(-N,N)
    
    X,Y = np.meshgrid(x,y)
    
    expFT = np.exp(((1j*k)/(2*z))*(((X*deltax0)**2)+((Y*deltay0)**2)))
    F= np.fft.fft2(Field*expFT*deltax0*deltay0)
    
    amp = np.exp(((1j*k)/(2*z))*((X*deltax)**2+(Y*deltay)**2))*(np.exp(1j*k*z)/(1j*w_length*z))
    #F= np.fft.fftshift(F)*amp
    F=F*amp
    
    return F
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
z=73000  #(73mm)
w_length=0.633   #All units in um
deltax0=3.18 
deltay0=3.18

deltax=(w_length*z)/(2*M*deltax0)
deltay=(w_length*z)/(2*M*deltay0)

print("deltax:",deltax)
print("deltay:",deltay)

lim=2*N*(deltax0**2)/w_length  #Limit of z in FT
print ("lim:",lim)
if z<lim:
    print("z limit exceeded")

tic=time.time()

T=Tmatrix(deltax0, deltay0, M,N)
holo= cv2.imread("Hologram.tiff",0)
FTholo= np.fft.fft2(holo)
FTholo= np.fft.fftshift(FTholo)
Filter=T*FTholo
InvFT=np.fft.ifft2(Filter)
Ref,FTRef=Ref(w_length, deltax0, deltay0, M,N)
U1=InvFT*Ref
Reconst=Fresnel(U1, z, deltax0, deltay0, w_length, deltax, deltay, M, N)

#Intensity
Iholo=np.log(np.abs(FTholo)**2)
IFilter=np.log((np.abs(Filter)**2)+0.0001)
IInvFT=np.abs(InvFT)**2
IR=np.log(np.abs(FTRef)**2)
IU1=np.log(np.abs(U1)**2)
IReconst=(np.abs(Reconst)**2)

#Figures
plt.imsave("Transmittance.png",T, cmap='gray') 
plt.imsave("Fourier Transform Hologram.png",Iholo, cmap='gray')   
plt.imsave("FT Hologram filtered.png",IFilter, cmap='gray')  
plt.imsave("Inverse FT filtered.png",IInvFT, cmap='gray')  
plt.imsave("FT Reference.png",IR, cmap='gray')  
plt.imsave("U+1.png",IU1, cmap='gray')  
plt.imsave("Reconstruction.png",IReconst, cmap='gray') 