import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import scipy.spatial.distance
import scipy.signal
import skimage
import skimage.io
from skimage.util import img_as_float
from scipy.ndimage.filters import median_filter

#Function for CLAHE
def histoeq(image):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(image)
	return cl1

#Function for Saturation Weights
def saturationWeight(image):
	b,g,r = cv2.split(image)
	lum = 0.2126*r + 0.7152*g + 0.0722*b
	w_sat = np.sqrt((((r-lum)**2)+((g-lum)**2)+((b-lum)**2))/3)
	return w_sat

#Function for Laplacian Contrast Weigths
def laplacianWeight(image):
	b,g,r = cv2.split(image)
	lum = 0.2126*r + 0.7152*g + 0.0722*b
	w_l = cv2.Laplacian(lum,cv2.CV_64F)
	w_l = np.absolute(w_l)
	return w_l

#Function for Saliency Weights
def saliencyWeight(image):
	img_rgb = img_as_float(image[:,:,::-1])
	img_lab = skimage.color.rgb2lab(img_rgb) 
	mean_val = np.mean(img_rgb,axis=(0,1))
	kernel_h = (1.0/16.0) * np.array([[1,4,6,4,1]])
	kernel_w = kernel_h.transpose()
	blurred_l = scipy.signal.convolve2d(img_lab[:,:,0],kernel_h,mode='same')
	blurred_a = scipy.signal.convolve2d(img_lab[:,:,1],kernel_h,mode='same')
	blurred_b = scipy.signal.convolve2d(img_lab[:,:,2],kernel_h,mode='same')
	blurred_l = scipy.signal.convolve2d(blurred_l,kernel_w,mode='same')
	blurred_a = scipy.signal.convolve2d(blurred_a,kernel_w,mode='same')
	blurred_b = scipy.signal.convolve2d(blurred_b,kernel_w,mode='same')
	im_blurred = np.dstack([blurred_l,blurred_a,blurred_b])
	sal = np.linalg.norm(mean_val - im_blurred,axis = 2)
	sal_max = np.max(sal)
	sal_min = np.min(sal)
	w_s = 255 * ((sal - sal_min) / (sal_max - sal_min))
	return w_s

#Function to generate 3 level Gaussian Pyramid
def GaussianPyramid(image):
	gaussian_pyr = [image]
	for i in range(3):
    		image = cv2.pyrDown(image)
    		gaussian_pyr.append(image)
	return gaussian_pyr

#Function to generate 3 level Laplacian Pyramid
def LaplacianPyramid(image):
	gaussian_pyr = GaussianPyramid(image)
	laplacian_top = gaussian_pyr[-1]
	laplacian_pyr = []
	for i in range(3):
		img = cv2.pyrUp(gaussian_pyr[i+1])
		dim = (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0])
		img = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
		laplacian = cv2.subtract(gaussian_pyr[i],img)
	    	laplacian_pyr.append(laplacian)
	laplacian_pyr.append(laplacian_top)
	return laplacian_pyr
	
def multiplyWeight(image,weight):
	b,g,r = cv2.split(image)
	scipy.signal.convolve2d(b,weight,mode="same")
	scipy.signal.convolve2d(g,weight,mode="same")
	scipy.signal.convolve2d(r,weight,mode="same")
	res = cv2.merge((b,g,r))
	return res

img = [cv2.imread(file) for file in glob.glob("*.jpeg")]
for i in range(0,len(img)):
	print(img[i].shape)
	length = img[i].shape[0]
	width = img[i].shape[1]
	factor = round(max(length/500.0,width/500.0))
	print(factor)
	image = cv2.resize(img[i],None,fx = 1/factor,fy = 1/factor)
	cv2.imwrite('Input/' + str(i+1) + '.jpeg',image)
	#CLAHE
	b,g,r = cv2.split(image)
	b = histoeq(b)
	g = histoeq(g)
	r = histoeq(r)
	img_hist_eq = cv2.merge((b,g,r))
	cv2.imwrite('CLAHE/'+ str(i+1) +'.jpeg',img_hist_eq)

	#Gray World Algorithm
	b_mean = cv2.mean(b)
	g_mean = cv2.mean(g)
	r_mean = cv2.mean(r)
	alpha = 0.6
	r_comp = r + alpha*(127 - r_mean[0])
	b_comp = b + alpha*(127 - b_mean[0])
	g_comp = g + alpha*(127 - g_mean[0])
	r_comp = np.clip(r_comp,0,255)
	g_comp = np.clip(g_comp,0,255)
	b_comp = np.clip(b_comp,0,255)
	img_wb = np.dstack((b_comp,g_comp,r_comp))
	cv2.imwrite('White Balance/'+ str(i+1) +'.jpeg',img_wb)
	
	#Input 1
	#Gamma Correction
	gamma = 1.2
	img_gamma = np.array(255*(img_wb / 255) ** gamma, dtype = 'uint8')

	#Noise Removal
	img_res = cv2.medianBlur(img_gamma,3)
	print(img_res.shape)
	cv2.imwrite('Gamma Correction/'+ str(i+1) +'.jpeg',img_res)
	
	#Saturation Weights
	ip1_w_sat = saturationWeight(img_res)
	cv2.imwrite('Input 1 Saturation Weights/'+ str(i+1) +'.jpeg',ip1_w_sat)

	#Laplacian Contrast Weights
	ip1_w_l = laplacianWeight(img_res)
	cv2.imwrite('Input 1 Lapacian Contrast Weights/'+ str(i+1) +'.jpeg',ip1_w_l)

	#Saliency Weights
	ip1_w_s = saliencyWeight(img_res)
	cv2.imwrite('Input 1 Saliency Weights/'+ str(i+1) +'.jpeg',ip1_w_s)

	#Input 2
	#Unsharp Masking
	img_Gaussian = cv2.GaussianBlur(img_wb, (0, 0), 10.0)
	img_unsharp = cv2.addWeighted(img_wb, 1.5, img_Gaussian, -0.5, 0, img_wb)
	cv2.imwrite('Sharpening/'+ str(i+1) +'.jpeg',img_unsharp)

	#Normalising
	img_normalised = cv2.normalize(img_unsharp, None, 0, 255, norm_type=cv2.NORM_MINMAX)
	print(img_normalised.shape)
	cv2.imwrite('Norm Sharpening/'+ str(i+1) +'.jpeg',img_normalised)

	#Saturation Weights
	ip2_w_sat = saturationWeight(img_normalised)
	cv2.imwrite('Input 2 Saturation Weights/'+ str(i+1) +'.jpeg',ip2_w_sat)

	#Laplacian Contrast Weights
	ip2_w_l = laplacianWeight(img_normalised)
	cv2.imwrite('Input 2 Lapacian Contrast Weights/'+ str(i+1) +'.jpeg',ip2_w_l)

	#Saliency Weights
	ip2_w_s = saliencyWeight(img_normalised)
	cv2.imwrite('Input 2 Saliency Weights/'+ str(i+1) +'.jpeg',ip2_w_s)
	
	#Aggregated Weights
	ip1_w_k = ip1_w_sat + ip1_w_l + ip1_w_s
	ip2_w_k = ip2_w_sat + ip2_w_l + ip2_w_s
	cv2.imwrite('Input 1 Aggregated Weights/'+ str(i+1) + '.jpeg',ip1_w_k)
	cv2.imwrite('Input 2 Aggregated Weights/'+ str(i+1) + '.jpeg',ip2_w_k)

	#Normalised Weights
	ip1_w_k_norm = ((ip1_w_k + 0.1)/(ip1_w_k + ip2_w_k + 0.2))*255
	ip2_w_k_norm = ((ip2_w_k + 0.1)/(ip1_w_k + ip2_w_k + 0.2))*255
	cv2.imwrite('Input 1 Normalised Weights/'+ str(i+1) + '.jpeg',ip1_w_k_norm)
	cv2.imwrite('Input 2 Normalised Weights/'+ str(i+1) + '.jpeg',ip2_w_k_norm)

	#Laplacian Pyramids of Fusion Inputs
	ip1_laplacian_pyr = LaplacianPyramid(img_res)
	ip2_laplacian_pyr = LaplacianPyramid(img_normalised)

	#Gaussian Pyramids of Normalised Weights
	ip1_w_k_norm_gaussian_pyr = GaussianPyramid(ip1_w_k_norm)
	ip2_w_k_norm_gaussian_pyr = GaussianPyramid(ip2_w_k_norm)

	for j in range(1,4):
		cv2.imwrite('Input 1 Gaussian Pyramid/' + str(i+1) + '_' + str(j) + '.jpeg',ip1_w_k_norm_gaussian_pyr[j])
		cv2.imwrite('Input 2 Gaussian Pyramid/' + str(i+1) + '_' + str(j) + '.jpeg',ip2_w_k_norm_gaussian_pyr[j])
		cv2.imwrite('Input 1 Laplacian Pyramid/' + str(i+1) + '_' + str(j) + '.jpeg',ip1_laplacian_pyr[j])
		cv2.imwrite('Input 2 Laplacian Pyramid/' + str(i+1) + '_' + str(j) + '.jpeg',ip2_laplacian_pyr[j])
	
	dim = (ip1_laplacian_pyr[1].shape[1], ip1_laplacian_pyr[1].shape[0])
	level_1_1 = multiplyWeight(ip1_laplacian_pyr[1],ip1_w_k_norm_gaussian_pyr[1])
	level_2_1 = cv2.pyrUp(multiplyWeight(ip1_laplacian_pyr[2],ip1_w_k_norm_gaussian_pyr[2]))
	level_2_1 = cv2.resize(level_2_1, dim, interpolation = cv2.INTER_NEAREST)
	level_3_1 = cv2.pyrUp(cv2.pyrUp(multiplyWeight(ip1_laplacian_pyr[3],ip1_w_k_norm_gaussian_pyr[3])))
	level_3_1 = cv2.resize(level_3_1, dim, interpolation = cv2.INTER_NEAREST)
	level_1_2 = multiplyWeight(ip2_laplacian_pyr[1],ip2_w_k_norm_gaussian_pyr[1])
	level_2_2 = cv2.pyrUp(multiplyWeight(ip2_laplacian_pyr[2],ip2_w_k_norm_gaussian_pyr[2]))
	level_2_2 = cv2.resize(level_2_2, dim, interpolation = cv2.INTER_NEAREST)
	level_3_2 = cv2.pyrUp(cv2.pyrUp(multiplyWeight(ip2_laplacian_pyr[3],ip2_w_k_norm_gaussian_pyr[3])))
	level_3_2 = cv2.resize(level_3_2, dim, interpolation = cv2.INTER_NEAREST)
	level_1 = level_1_1 + level_1_2
	level_2 = level_2_1 + level_2_2
	level_3 = level_3_1 + level_3_2
	result = cv2.add(cv2.add(level_1,level_2),level_3)
	result = cv2.normalize(result, None, 0, 255, norm_type=cv2.NORM_MINMAX)
	print(i)
	cv2.imwrite('Results/'+ str(i+1) + '.jpeg',result)
