import numpy as np
import multiprocessing
import threading
import queue
import imageio
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers

def build_recognition_system(vgg16,num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,K)
	* labels: numpy.ndarray of shape (N)
	'''
	trained_system_path = 'trained_system_deep.npz'
	
	if not os.path.exists(trained_system_path):
		print('building system from train data')

		train_data = np.load("../data/train_data.npz")
		
		train_labels = train_data['labels']
		train_image_names = train_data['image_names']
		data_dir = '../data'

		device = 'cpu'
		if torch.cuda.is_available():
			device = 'cuda'
			vgg16.to(device)

		# exract train set feature
		train_features = list()
		for image_idx, image_name in enumerate(train_image_names):
			image_path = os.path.join(data_dir, image_name[0])
			image_feat = get_image_feature((image_idx, image_path, vgg16, device))
			train_features.append(image_feat)
		train_features = np.stack(train_features, axis=0)
		
		# save trained system
		np.savez(trained_system_path, features=train_features, labels=train_labels)
	
	print('trained system created.')


def evaluate_recognition_system(vgg16,num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	test_data = np.load("../data/test_data.npz")
	trained_system = np.load("trained_system_deep.npz")
	
	# unzip test data
	test_labels = test_data['labels']
	test_image_names = test_data['image_names']
	data_dir = '../data'
	
	# unzip trained system
	features = trained_system['features']
	labels = trained_system['labels']

	test_features_path = 'test_features_vgg.npy'
	if not os.path.exists(test_features_path):
		# exract test set feature
		print('extracting test set feature')
		test_features = []
		
		device = 'cpu'
		if torch.cuda.is_available():
			device = 'cuda'
			vgg16.to(device)
		
		for image_idx, image_name in enumerate(test_image_names):
			image_path = os.path.join(data_dir, image_name[0])
			image_feat = get_image_feature((image_idx, image_path, vgg16, device))
			test_features.append(image_feat)
		test_features = np.stack(test_features, axis=0)
		np.save(test_features_path, test_features)
		print('done')
	else:
		print('reading test features from cache')
		test_features = np.load(test_features_path)

	# predict class label and build conf matrix
	conf = np.zeros((8, 8))
	for test_idx, test_feature in enumerate(test_features):
		distances = distance_to_set(test_feature, features)
		pred_cls = labels[np.argmax(distances)]
		conf[test_labels[test_idx], pred_cls] += 1
	
	accuracy = np.trace(conf) / np.sum(conf)

	return conf, accuracy


def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: torch.Tensor of shape (1,3,H,W)
	'''
	image = network_layers.preprocess(image, (224, 224))
	image_processed = network_layers.to_pytorch(image)
	return image_processed
	

def get_image_feature(args):
	'''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
 	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.
	* time_start: time stamp of start time
 	[saved]
	* feat: evaluated deep feature
	'''

	t0 = time.time()
	i, image_path, vgg16, device = args
	image = skimage.io.imread(image_path)
	image = network_layers.preprocess(image, (224, 224))
	feat = network_layers.extract_deep_feature_pytorch(image, vgg16, device=device)
	t1 = time.time()
	print('processed image %d in %fs' % (i, t1-t0))
	return feat


def distance_to_set(feature,train_features):
	'''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N,K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''

	dist = -1*np.sum((train_features - feature)**2, axis=1)
	return dist