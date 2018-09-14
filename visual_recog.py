import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words

def build_recognition_system(num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,M)
	* labels: numpy.ndarray of shape (N)
	* dictionary: numpy.ndarray of shape (K,3F)
	* SPM_layer_num: number of spatial pyramid layers
	'''



	train_data = np.load("../data/train_data.npz")
	dictionary = np.load("dictionary.npy")
	# ----- TODO -----

	pass

def evaluate_recognition_system(num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''


	test_data = np.load("../data/test_data.npz")
	trained_system = np.load("trained_system.npz")
	# ----- TODO -----
	pass




def get_image_feature(file_path,dictionary,layer_num,K):
	'''
	Extracts the spatial pyramid matching feature.

	[input]
	* file_path: path of image file to read
	* dictionary: numpy.ndarray of shape (K,3F)
	* layer_num: number of spatial pyramid layers
	* K: number of clusters for the word maps

	[output]
	* feature: numpy.ndarray of shape (K)
	'''
	pass


	# ----- TODO -----


def distance_to_set(word_hist,histograms):
	'''
	Compute similarity between a histogram of visual words with all training image histograms.

	[input]
	* word_hist: numpy.ndarray of shape (K)
	* histograms: numpy.ndarray of shape (N,K)

	[output]
	* sim: numpy.ndarray of shape (N)
	'''
	pass
	


	# ----- TODO -----



def get_feature_from_wordmap(wordmap,dict_size, norm=False):
	'''
	Compute histogram of visual words.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* dict_size: dictionary size K

	[output]
	* hist: numpy.ndarray of shape (K)
	'''
	hist, bin_edge = np.histogram(wordmap, bins=dict_size, range=(0, dict_size))
	if norm:
		hist = hist / np.sum(hist)
	return hist



def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
	'''
	Compute histogram of visual words using spatial pyramid matching.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* layer_num: number of spatial pyramid layers
	* dict_size: dictionary size K

	[output]
	* hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
	'''
	# compute finest hists
	img_h, img_w = wordmap.shape[0], wordmap.shape[1]
	finest_splits = 2**(layer_num-1)
	
	cell_h = int(np.ceil(img_h / finest_splits))
	cell_w = int(np.ceil(img_w / finest_splits))
	finest_hists = list()
	for i in range(finest_splits):
		for j in range(finest_splits):
			cur_cell = wordmap[i*cell_h:min((i+1)*cell_h, img_h), j*cell_w:min((j+1)*cell_w, img_w)]
			cur_cell_hist = get_feature_from_wordmap(cur_cell, dict_size, norm=False)
			finest_hists.append(cur_cell_hist)
	
	pyramid_hists = [finest_hists]
	for l in range(layer_num-1):
		cur_layer_hist = list()
		cur_layer_splits = int(finest_splits / (2**(l+1)))
		prev_layer_splits = cur_layer_splits*2
		prev_hist = pyramid_hists[-1]
		for i in range(cur_layer_splits):
			for j in range(cur_layer_splits):
				# 0 1
				# 2 3
				h0 = prev_hist[2*i*prev_layer_splits + 2*j]
				h1 = prev_hist[2*i*prev_layer_splits + 2*j+1]
				h2 = prev_hist[(2*i+1)*prev_layer_splits + 2*j]
				h3 = prev_hist[(2*i+1)*prev_layer_splits + 2*j+1]
				cur_cell_hist = h0+h1+h2+h3
				cur_layer_hist.append(cur_cell_hist)
		pyramid_hists.append(cur_layer_hist)
	
	all_hists = list()
	for layer_idx, layer in enumerate(pyramid_hists):
		if layer_idx == len(pyramid_hists)-1:
			weight = 0.5**layer_idx
		else:
			weight = 0.5**(layer_idx+1)
		
		for hist in layer:
			weighted_hist = (hist / (img_h*img_w)) * weight
			all_hists.append(weighted_hist)

	feature_vec = np.concatenate(all_hists, axis=0)
	return feature_vec





	

