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

def Pupila(w_l, dx0, N):
    """
    Incident wave and transmittance function
    In this case: plane wave and circular aperture 
    """
    na=0.25
    dx0=w_l*20*(1e3)/(4*N*dx0)
    dy0=dx0
    N=2*N
    x=np.arange(-N,N)
    y=np.arange(-N,N)
    x,y=np.meshgrid(x,y)
    x=x*dx0
    y=y*dy0
    lim=na*2*1e4/np.sqrt(1-0.25*0.25)      #20000 um= 20 mm
#    lim=2*1e5
    U_matrix=(x)**2 + (y)**2
    U_matrix[np.where(np.abs(U_matrix)<=lim)]=1
    U_matrix[np.where(np.abs(U_matrix)>lim)]=0
    print (lim, 512*w_l*200*1e3)
    return U_matrix

def Imagen(w_l, dx0, N):
    """
    Incident wave and transmittance function
    In this case: plane wave and circular aperture 
    """
    #dx0=dx0.     #Pixeles 10 veces más pequeños. dx0=2.5 um
    N=2*N
    dy0=dx0
    x=np.arange(-N,N)
    y=np.arange(-N,N)
    x,y=np.meshgrid(x,y)

    lim=1e4         #Tamaño 10 mm
    U_matrix=(x*dx0)**2 + (y*dy0)**2
    U_matrix[np.where(U_matrix<=lim)]=1
    U_matrix[np.where(U_matrix>lim)]=0
#    print (lim, 512*w_l*200*1e3)
    return U_matrix


w_l=0.533          #(533nm)
dx0=2.5             #2.5um tamaño de pixel
N=M=int(512/2)     #Number of pixels


r=int(512/2)

"Para el cameraman"
U_0=cv2.imread('cameraman.png',0)
U_0 = cv2.copyMakeBorder(U_0,r,r,r,r,cv2.BORDER_CONSTANT)

"Para el circulo"
#U_0=Imagen(w_l, dx0, N) 
   




#U_1=np.fft.fft2(U_0)
P=Pupila(w_l, dx0, N)
P_FT=np.fft.fft2(P/(w_l*2*1e5))

U_1=0.1*np.fft.fftshift(np.fft.fftn(0.1*U_0*(dx0)**2))

x=y=2*N*dx0
#x_0=y_0=0.1*x


u=np.arange(-N,N)
v=np.arange(-N,N)
u,v=np.meshgrid(u,v)
mascara=u*(dx0*0.1)+v*dx0*0.1
mascara[:]=1.
mascara=cv2.copyMakeBorder(mascara,r,r,r,r,cv2.BORDER_CONSTANT)


Uf=U_1*P

#Uf=Uf*mascara

Uf=(np.fft.fftn(Uf*dx0**2))
I1=np.log(np.abs(U_1)**2)
I=(np.abs(Uf)**2) 
angle=np.angle(Uf) 

plt.figure(1)
plt.imshow(U_0, cmap='gray',extent=[-x,x,-y,y])

plt.figure(2)
plt.imshow(P, cmap='gray')

plt.figure(3)
plt.imshow(angle, cmap='gray')

plt.figure(4)
plt.imshow(I, cmap='gray', extent=[-x,x,-y,y])
plt.imsave("Final-cameraman.png",I, cmap='gray')
    
    
    
    
    
    

    