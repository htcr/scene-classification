import numpy as np
import scipy.ndimage
import os,time
import util
import torch
from visual_words import to_3channel
import skimage
import skimage.transform

def extract_deep_feature(x,vgg16_weights):
	'''
	Extracts deep features from the given VGG-16 weights.

	[input]
	* x: numpy.ndarray of shape (H,W,3)
	* vgg16_weights: numpy.ndarray of shape (L,3)

	[output]
	* feat: numpy.ndarray of shape (K)
	'''
	
	met_fc = False
	for layer in vgg16_weights:
		name = layer[0]
		if name == 'conv2d':
			x = multichannel_conv2d(x, layer[1], layer[2])
		elif name == 'relu':
			x = relu(x)
		elif name == 'maxpool2d':
			x = max_pool2d(x, layer[1])
		elif name == 'linear':
			if not met_fc:
				x = x.transpose([2, 0, 1]).reshape(-1)
				met_fc = True
			x = linear(x, layer[1], layer[2])
	
	return x

def preprocess(image, size):
	image = skimage.transform.resize(image, size, mode='constant')
	image = to_3channel(image)
	image.astype(np.float32)/255
	mean = np.array([0.485,0.456,0.406], dtype=np.float32)[np.newaxis, np.newaxis, :]
	std = np.array([0.229,0.224,0.225], dtype=np.float32)[np.newaxis, np.newaxis, :]
	image = (image-mean) / std
	return image

def to_pytorch(x):
	'''
	Preprocess image to pytorch-acceptable tensor
	[input]
	* x: numpy.ndarray of shape (H,W,3)
	[output]
	* tensor: pytorch tensor [N, C, H, W]
	'''
	x = np.transpose(x, [2, 0, 1])[np.newaxis, :, :]
	tensor = torch.Tensor(x)
	return tensor
	
def extract_deep_feature_pytorch(x, model, device='cpu'):
	'''
	Extracts deep features from VGG-16 using pytorch

	[input]
	* x: numpy.ndarray of shape (H,W,3)
	* model: pytorch model

	[output]
	* feat: numpy.ndarray of shape (K)
	'''
	tensor = to_pytorch(x).to(device)
	output = model(tensor)
	feat = output.detach().to('cpu').numpy().reshape(-1)
	return feat

def multichannel_conv2d(x,weight,bias):
	'''
	Performs multi-channel 2D convolution.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* feat: numpy.ndarray of shape (H,W,output_dim)
	'''
	c_in = x.shape[2]
	c_out = weight.shape[0]
	h, w = x.shape[0], x.shape[1]
	feat = np.zeros((h, w, c_out), dtype=np.float32)

	for i in range(c_out):
		for j in range(c_in):
			feat[:, :, i] += scipy.ndimage.correlate(x[:, :, j], weight[i, j, :, :], mode='constant')
		feat[:, :, i] += bias[i]
	
	return feat
	

def relu(x):
	'''
	Rectified linear unit.

	[input]
	* x: numpy.ndarray

	[output]
	* y: numpy.ndarray
	'''
	return x*((x>0).astype(np.float32))
	

def max_pool2d(x,size):
	'''
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* size: pooling receptive field

	[output]
	* y: numpy.ndarray of shape (H/size,W/size,input_dim)
	'''
	h_in, w_in, c = x.shape
	h_in, w_in = h_in - (h_in % size), w_in - (w_in % size)

	h_out, w_out = int(h_in / size), int(w_in / size)

	h_idx = np.arange(h_in)
	h_idx = np.repeat(h_idx, size)
	h_idx = h_idx.reshape(-1, size*size)
	h_idx = np.repeat(h_idx, w_out, axis=0)
	h_idx = h_idx.reshape(1, -1)
	h_idx = np.repeat(h_idx, c, axis=0)
	h_idx = h_idx.reshape(-1)

	w_idx = np.arange(w_in)
	w_idx = w_idx.reshape(-1, size)
	w_idx = w_idx.repeat(size, axis=0)
	w_idx = w_idx.reshape(1, -1)
	w_idx = w_idx.repeat(h_out, axis=0)
	w_idx = w_idx.reshape(1, -1)
	w_idx = w_idx.repeat(c, axis=0)
	w_idx = w_idx.reshape(-1)

	c_idx = np.arange(c)
	c_idx = c_idx.repeat(h_in*w_in)

	x_reindex = x[h_idx, w_idx, c_idx]
	x_reindex = x_reindex.reshape(-1, size*size)
	y_reindex = np.max(x_reindex, axis=1)
	y_reindex = y_reindex.reshape(c, h_out, w_out)
	y = np.transpose(y_reindex, [1, 2, 0])

	return y

def linear(x,W,b):
	'''
	Fully-connected layer.

	[input]
	* x: numpy.ndarray of shape (input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* y: numpy.ndarray of shape (output_dim)
	'''
	
	return W @ x + b


def test_conv():
	vgg16_weights = util.get_VGG16_weights()
	x = np.random.rand(227, 227, 3).astype(np.float32)
	weight = vgg16_weights[0][1]
	bias = vgg16_weights[0][2]
	response_mine = multichannel_conv2d(x, weight, bias)
	
	t_x = torch.Tensor(np.transpose(x, [2, 0, 1])[np.newaxis, :, :, :])
	t_w = torch.Tensor(weight)
	t_b = torch.Tensor(bias)
	t_response_pt = torch.nn.functional.conv2d(t_x, t_w, t_b, padding=1)
	response_pt = t_response_pt.numpy()
	response_pt = np.transpose(response_pt[0, :, :, :], [1, 2, 0])

	error = np.sum(np.abs(response_mine-response_pt)) / np.sum(np.abs(response_pt))
	print('conv: %f' % error)

def test_pool():
	x = np.random.rand(227, 227, 3).astype(np.float32)
	ksize = 3
	response_mine = max_pool2d(x, ksize)
	
	t_x = torch.Tensor(np.transpose(x, [2, 0, 1])[np.newaxis, :, :, :])
	t_response_pt = torch.nn.functional.max_pool2d(t_x, kernel_size=ksize, stride=ksize)
	response_pt = t_response_pt.numpy()
	response_pt = np.transpose(response_pt[0, :, :, :], [1, 2, 0])

	error = np.sum(np.abs(response_mine-response_pt)) / np.sum(np.abs(response_pt))
	print('pool: %f' % error)

def test_relu():
	x = np.random.randn(227, 227, 3).astype(np.float32)
	response_mine = relu(x)
	
	t_x = torch.Tensor(np.transpose(x, [2, 0, 1])[np.newaxis, :, :, :])
	t_response_pt = torch.nn.functional.relu(t_x)
	response_pt = t_response_pt.numpy()
	response_pt = np.transpose(response_pt[0, :, :, :], [1, 2, 0])

	error = np.sum(np.abs(response_mine-response_pt)) / np.sum(np.abs(response_pt))
	print('relu: %f' % error)

def test_linear():
	x = np.random.randn(100).astype(np.float32)
	weight = np.random.randn(200, 100)
	bias = np.random.randn(200)
	response_mine = linear(x, weight, bias)
	
	t_x = torch.Tensor(x[np.newaxis, :])
	t_w = torch.Tensor(weight)
	t_b = torch.Tensor(bias)
	t_response_pt = torch.nn.functional.linear(t_x, t_w, t_b)
	response_pt = t_response_pt.numpy()
	response_pt = response_pt[0, :]

	error = np.sum(np.abs(response_mine-response_pt)) / np.sum(np.abs(response_pt))
	print('linear: %f' % error)

def test_components():
	print('component error:')
	test_conv()
	test_pool()
	test_relu()
	test_linear()

def test_vgg16():
	vgg16_fc7 = util.vgg16_fc7()
	vgg16_fc7_weights = util.get_VGG_weights(vgg16_fc7)
	x = np.random.randn(500, 300, 3)
	
	x = preprocess(x, (224, 224))
	out_mine = extract_deep_feature(x, vgg16_fc7_weights)
	vgg16_fc7.to('cuda')
	out_pt = extract_deep_feature_pytorch(x, vgg16_fc7, device='cuda')
	
	print(out_mine.shape)
	print(out_pt.shape)
	error = np.sum(np.abs(out_mine-out_pt)) / np.sum(np.abs(out_pt))
	print('vgg error: %f' % error)
	
if __name__=='__main__':
	# use pytorch functions as baseline.
	# test_components()
	test_vgg16()