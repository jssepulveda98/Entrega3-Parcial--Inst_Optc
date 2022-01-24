import numpy as np
import matplotlib.pyplot as plt
import cv2


"""
    Scheme
    1. Discretize the input image
    2. Takking the fft 
    3. Simulate the pupil function or system H function 
    4. Make the convolution of the FT image with the H function
    
    
"""



def Pupila(w_l, dx0, N,Prop):
    """
    Incident wave and transmittance function
    In this case: plane wave and circular aperture 
    """
    
    f=1/((1/15e3) -(1/Prop))
    na=(1e3)/np.sqrt(f**2 +(1e3)**2)
    dx0=w_l*Prop/(dx0*N)
    dy0=dx0
    N=N
    x=np.arange(-N,N)
    y=np.arange(-N,N)
    x,y=np.meshgrid(x,y)
    x=x*dx0
    y=y*dy0
    lim=2000*na     

    U_matrix=(x)**2 + (y)**2
    U_matrix[np.where(np.abs(U_matrix)<=lim)]=1
    U_matrix[np.where(np.abs(U_matrix)>lim)]=0
    
    return U_matrix

def Zeros(Image):
    N=max(np.shape(Image))
    if N%2 != 0:
        N=N+1
        
    Zeros=np.zeros((2*N,2*N))
    n,m=np.shape(Image)[0],np.shape(Image)[1]
    Zeros[int(N -n/2): int(N +n/2),int(N -m/2): int(N+m/2)]=Image
    
    return Zeros

def Form(Image,ld,dx0,Prop):
    Im=Zeros(Image)
    n,m=np.shape(Image)[0],np.shape(Image)[1]
    N=max(np.shape(Image))
    Pu=Pupila(ld,dx,2836,Prop)
    
    Im=(15e3/Prop)*np.fft.fftshift(np.fft.fftn((15e3/Prop)*Im *(dx0)**2))
    P=(np.fft.fftn(Pu/(ld*Prop)))
    
    Im=(np.fft.ifftn(Im*Pu))
    
    
    I=np.abs(Im[int(N -n/2): int(N +n/2),int(N -m/2): int(N+m/2)])**2
    
    
    plt.figure()
    plt.imshow(I,cmap="gray")
    plt.imsave("Prop %d m.jpg" %(Prop/1e6),I,cmap="gray")




image=cv2.imread("gala-desnuda-mirando.jpg",0)

dx=24e4
Prop=1e6*np.array([1,100,200,300,500])

for i in Prop:
    Form(image,0.700,dx,i)
    
    
    
    
    

    