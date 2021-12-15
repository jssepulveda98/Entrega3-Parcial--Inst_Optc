import matplotlib.pyplot as plt
import numpy as np
import time
import cv2



holo= cv2.imread("Hologram.tiff",0)
FTholo= np.fft.fft2(holo)
FTholo= np.fft.fftshift(FTholo)
Iholo=np.log(np.abs(FTholo)**2)

plt.imsave("Fourier Transform Hologram.png",Iholo, cmap='gray')   