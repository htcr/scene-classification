import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import matplotlib.pyplot as plt
import util
import random

from scipy.ndimage import gaussian_filter, gaussian_laplace

def extract_filter_responses(image):
	'''
	Extracts the filter responses for the given image.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	[output]
	* filter_responses: numpy.ndarray of shape (H,W,3F)
	'''
	# handle non 3-channel images
	if len(image.shape) < 3:
		image = image[:, :, np.newaxis]
	
	if image.shape[2] == 1:
		image = np.concatenate([image, image, image], axis=2)
	elif image.shape[2] > 3:
		image = image[:, :, 0:3]
	
	assert len(image.shape) == 3 and image.shape[2] == 3

	# filter sigmas
	sigmas = [1, 2, 4, 8, 11.3137]
	output = np.zeros((image.shape[0], image.shape[1], 3*20), dtype=np.float32)
	for sigma_idx, sigma in enumerate(sigmas):
		for ch_idx in range(3):
			output[:, :, sigma_idx*4*3 + ch_idx] = gaussian_filter(image[:, :, ch_idx], sigma, mode='constant')
			output[:, :, sigma_idx*4*3 + 3 + ch_idx] = gaussian_laplace(image[:, :, ch_idx], sigma, mode='constant')
			output[:, :, sigma_idx*4*3 + 6 + ch_idx] = gaussian_filter(image[:, :, ch_idx], sigma, order=[0, 1], mode='constant')
			output[:, :, sigma_idx*4*3 + 9 + ch_idx] = gaussian_filter(image[:, :, ch_idx], sigma, order=[1, 0], mode='constant')
	
	return output


def get_visual_words(image,dictionary):
	'''
	Compute visual words mapping for the given image using the dictionary of visual words.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	
	[output]
	* wordmap: numpy.ndarray of shape (H,W)
	'''
	
	# ----- TODO -----
	pass


def compute_dictionary_one_image(args):
	'''
	Extracts random samples of the dictionary entries from an image.
	This is a function run by a subprocess.

	[input]
	* i: index of training image
	* alpha: number of random samples
	* image_path: path of image file
	* time_start: time stamp of start time

	[saved]
	* sampled_response: numpy.ndarray of shape (alpha,3F)
	'''


	i,alpha,image_path = args
	# ----- TODO -----
	pass


def compute_dictionary(num_workers=2):
	'''
	Creates the dictionary of visual words by clustering using k-means.

	[input]
	* num_workers: number of workers to process in parallel
	
	[saved]
	* dictionary: numpy.ndarray of shape (K,3F)
	'''

	train_data = np.load("../data/train_data.npz")
	# ----- TODO -----
	pass


