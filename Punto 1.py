import numpy as np
import matplotlib.pyplot as plt
import cv2


"""
    Scheme
    1. Discretize the input image
    2. Takking the fft 
    3. Simulate the pupil function or system H function 
    4. Make the convolution of the FT image with the H function
    
    f1=20 mm, f2=200 mm, r=20*0.25
"""



def Pupila(w_l, dx0, N):
    """
    Incident wave and transmittance function
    In this case: plane wave and circular aperture 
    """
    na=9.4E-6
    dx0=0.7
    dy0=dx0
    N=N/2
    x=np.arange(-N,N)
    y=np.arange(-N,N)
    x,y=np.meshgrid(x,y)
    x=x*dx0
    y=y*dy0
    lim=2000#2*1e4/np.sqrt(1-0.25*0.25)      #20000 um= 20 mm
#    lim=2*1e5
    U_matrix=(x)**2 + (y)**2
    U_matrix[np.where(np.abs(U_matrix)<=lim)]=1
    U_matrix[np.where(np.abs(U_matrix)>lim)]=0
    
    return U_matrix

def Zeros(Image):
    N=max(np.shape(Image))
    if N%2 != 0:
        N=N+1
        
    Zeros=np.zeros((N,N))
    print(np.shape(Zeros))
    Zeros[0:np.shape(Image)[0],0:np.shape(Image)[1]]=Image
    return Zeros

def Form(Image,ld):
    Im=Zeros(Image)
    
    P=Pupila(ld,2.5,2836)
    
    Im=np.fft.fftshift(np.fft.fftn((15E-3/600)*Im))
    P=(1/(ld**2  *600*15*1e-3))*(np.fft.fftn(P/(ld*600*1e6)))
    
    Im=np.fft.fftshift(np.fft.ifftn(Im*P))
    
    plt.imshow(np.abs(Im)**2,cmap="gray")




image=cv2.imread("gala-desnuda-mirando.jpg",0)



Form(image,700)
    
    
    
    
    

    