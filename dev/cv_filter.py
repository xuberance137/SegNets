import cv2
import numpy as numpy
import matplotlib.pyplot as plt

def cv_fft(img):
	dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
	dft_shift = np.fft.fftshift(dft)
	r = dft_shift[:,:,0]
	i = dft_shift[:,:,1]
	mag_spectrum = 20*np.log(cv2.mag_spectrum(r,i))
	return mag_spectrum

def create_mask(img):
	row, col = img.shape
	crow, ccol = row/2, col/2
	# LPF
	mask = np.zeros((row, col), np.uint8)
	mask[crow-20:crow+20, ccol-20:ccol+20] = 1
	# HPF
	# mask = np.ones((row, col), np.uint8)
	# mask[crow-20:crow+20, ccol-20:ccol+20] = 0
	return mask

def apply_mask(mask, img):
	dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
	dft_shift = np.fft.fftshift(dft)
	fshift = dft_shift*mask
	f_ishift = np.fft.ifftshift(fshift)
	img_recon = cv2.idft(f_ishift)
	img_recon = cv2.magnitude(img_recon[:,:,0], img_recon[:,:,1])
	return img_recon
