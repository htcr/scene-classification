import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import skimage.io

import network_layers
import torch

if __name__ == '__main__':

	num_cores = util.get_num_CPU()
	
	# skimage.io.imshow(image)
	# skimage.io.show()
	
	#path_img = "../data/auditorium/sun_aflgfyywvxbpeyxl.jpg"
	#path_img = "../data/baseball_field/sun_aalztykafqwxrspj.jpg"
	path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
	#path_img = "../data/highway/sun_acpvugnkzrliaqir.jpg"
	image = skimage.io.imread(path_img)
	image = image.astype('float')/255
	
	filter_responses = visual_words.extract_filter_responses(image)
	util.display_filter_responses(filter_responses)
	
	visual_words.compute_dictionary(num_workers=num_cores)	

	dictionary = np.load('dictionary.npy')
	wordmap = visual_words.get_visual_words(image,dictionary)
	util.save_wordmap(wordmap, 'word_map.jpg')
	

	visual_recog.build_recognition_system(num_workers=num_cores)

	conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
	print(conf)
	print(accuracy)

	#vgg16 = torchvision.models.vgg16(pretrained=True).double()
	#vgg16.eval()
	vgg16 = util.vgg16_fc7()

	#deep_recog.build_recognition_system(vgg16,num_workers=num_cores//2)
	conf, accuracy = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores//2)
	print(conf)
	print(accuracy)
