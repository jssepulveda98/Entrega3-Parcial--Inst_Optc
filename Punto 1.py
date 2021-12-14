import numpy as np
import matplotlib.pyplot as plt
import cv2


"""
    ESQUEMA
    1. Discretizar la imagen de entrada
    2. Calcular la TF de la imagen magnificada
    3. Programar la funcion h
    4. Multiplicar la TF de h con la TF de la imagen
    
    f1=20 mm, f2=200 mm, r=20*0.25
"""

def Pupila(z, w_l, dx0, N):
    """
    Incident wave and transmittance function
    In this case: plane wave and circular aperture 
    """
    na=0.25
    dy0=dx0
    x=np.arange(-N/2,N/2)
    y=np.arange(-N/2,N/2)
    x,y=np.meshgrid(x,y)
#    Nzones=20       #Number of Fresnel zones
#    lim=Nzones*w_l*z
    lim=0.25*20*dx0
    U_matrix=(x*dx0)**2 + (y*dy0)**2
    U_matrix[np.where(U_matrix<=lim)]=1
    U_matrix[np.where(U_matrix>lim)]=0

    return U_matrix

def FT(Uin, w_l, f1, f2, dx0, z):
    "-----Step 1------"
    k=2*np.pi/w_l
    N,M=np.shape(Uin)
    x=np.arange(-N/2,N/2,1)
    y=np.arange(-M/2,M/2,1)
    X,Y=np.meshgrid(x,y)
    #phase=np.exp((1j*k)/(2*z)*(((X*dx0)**2) + ((Y*dx0)**2)))
    U1=Uin*f1/f2
    "-----Step 2-----"
    X=X*(1/(M*dx0))
    Y=Y*(1/(N*dx0))
    Uf=np.fft.fftshift(np.fft.fft2(U1))
    "-----Step 3-----"

    #c1=np.exp(1j*k*z)/(1j*w_l*z)
    #Uf=Uf*c1*(np.exp((1j*(k/2*z))*((X*dx)**2 + (Y*dx)**2)))
    
    return Uf


U_0=cv2.imread('cameraman.png',0)
    
    
    
    
    
    
    
    

    