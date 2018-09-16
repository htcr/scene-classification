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
from multiprocessing import Pool

def to_3channel(image):
	# handle non 3-channel images
	if len(image.shape) < 3:
		image = image[:, :, np.newaxis]
	
	if image.shape[2] == 1:
		image = np.concatenate([image, image, image], axis=2)
	elif image.shape[2] > 3:
		image = image[:, :, 0:3]
	
	assert len(image.shape) == 3 and image.shape[2] == 3
	return image

def extract_filter_responses(image):
	'''
	Extracts the filter responses for the given image.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	[output]
	* filter_responses: numpy.ndarray of shape (H,W,3F)
	'''
	# handle non 3-channel images
	image = to_3channel(image)

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
	img_h, img_w = image.shape[0], image.shape[1]
	# get filter response
	filter_response = extract_filter_responses(image)
	feature_size = filter_response.shape[2]
	filter_response = filter_response.reshape((-1, feature_size))
	
	# compute distance
	pairwise_dist = scipy.spatial.distance.cdist(filter_response, dictionary)
	# pick nearest dictionary entry
	wordmap = np.argmin(pairwise_dist, axis=1)
	wordmap = wordmap.reshape((img_h, img_w))
	
	return wordmap


def compute_dictionary_one_image(args):
	'''
	Extracts random samples of the dictionary entries from an image.
	This is a function run by a subprocess.

	[input]
	* args[0] data_dir
	* args[1] image_name
	* args[2] feature_dir
	* args[3] alpha

	[saved]
	* sampled_response: numpy.ndarray of shape (alpha,3F)
	'''

	t0 = time.time()
	pid = os.getpid()
	data_dir, image_name, feature_dir, alpha = args
	
	image_path = os.path.join(data_dir, image_name)
	image = skimage.io.imread(image_path)
	image = image.astype('float')/255
	filter_responses = extract_filter_responses(image)
	
	feature_size = filter_responses.shape[2]
	filter_responses = filter_responses.reshape((-1, feature_size))
	
	pixel_num = filter_responses.shape[0]
	sampled_features = filter_responses[np.random.choice(pixel_num, alpha, replace=False), :]
	
	save_path = os.path.join(feature_dir, image_name.replace('/', '_'))
	np.save(save_path, sampled_features)
	
	t1 = time.time()
	print('p%d done with %s in %f seconds' % (pid, image_name, t1-t0))


def compute_dictionary(num_workers=2):
	'''
	Creates the dictionary of visual words by clustering using k-means.

	[input]
	* num_workers: number of workers to process in parallel
	
	[saved]
	* dictionary: numpy.ndarray of shape (K,3F)
	'''

	train_data = np.load("../data/train_data.npz")
	train_labels = train_data['labels']
	train_image_names = train_data['image_names']
	data_dir = '../data'
	feature_dir = '../feature'
	
	alpha = 50
	K = 100
	
	if not os.path.exists(feature_dir):
		os.makedirs(feature_dir)

		# construct argument list for multiprocessing
		args_ary = []
		for name in train_image_names:
			args_ary.append((data_dir, name[0], feature_dir, alpha))
		# run multiprocessing
		pool = Pool(processes=num_workers)
		pool.map(compute_dictionary_one_image, args_ary)
		print('done, all features saved')
	else:
		print('using cached features')
	
	# do clustering
	dictionary_path = 'dictionary.npy'
	if not os.path.exists(dictionary_path):
		# load features
		all_image_feature = list()
		image_num = len(train_image_names)
		for idx, image_name in enumerate(train_image_names):
			feature_path = os.path.join(feature_dir, image_name[0].replace('/', '_')+'.npy')
			image_feature = np.load(feature_path)
			all_image_feature.append(image_feature)
			print('%d/%d loaded' % (idx+1, image_num))
		print('concatenating...')
		all_image_feature = np.concatenate(all_image_feature, axis=0)
		print('all feature concatenated')

		print('doing KMeans')
		t0 = time.time()
		kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs=-1).fit(all_image_feature)
		t1 = time.time()
		print('KMeans done. Time elapsed: %f' % (t1-t0))
		dictionary = kmeans.cluster_centers_
		np.save(dictionary_path, dictionary)
	else:
		print('loading dictionary from cache')
		dictionary = np.load(dictionary_path)
	

