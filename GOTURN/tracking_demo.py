import sys

sys.path.append('CAFFEPATH/python')

import math
import numpy as np 
import cv2
import caffe
import datetime

from PIL import Image, ImageDraw, ImageFont
from BoundingBox import BoundingBox

#---------------------common setting----------------------
gpu_mode = False
gpu_id = 0
net_deploy = './models/tracker.prototxt'
model_weights = './models/tracker.caffemodel'
mean_value = [104, 117, 123]

if gpu_mode:
	caffe.set_mode_gpu()
	caffe.set_device(gpu_id)
else:
	caffe.set_mode_cpu()

t1 = datetime.datetime.now()
net = caffe.Net(net_deploy, model_weights, caffe.TEST)
t2 = datetime.datetime.now()
print 'load model:' ,t2 - t1

#----------------------network parameter---------------------
num_inputs = net.blobs['image'].data[...].shape[0]
channels = net.blobs['image'].data[...].shape[1]
height = net.blobs['image'].data[...].shape[2]
width = net.blobs['image'].data[...].shape[3]
print num_inputs, channels, height, width

#----------------------visualize parameter-------------------
file_pth = '.../Arial.ttf'
FONT10 = ImageFont.truetype(file_pth + '/../data/Arial.ttf', 10)
FONT15 = ImageFont.truetype(file_pth + '/../data/Arial.ttf', 15)
FONT20 = ImageFont.truetype(file_pth + '/../data/Arial.ttf', 20)
FONT30 = ImageFont.truetype(file_pth + '/../data/Arial.ttf', 30)
CVFONT0 = cv2.FONT_HERSHEY_SIMPLEX


def main(video_file, bboxes):
	video = cv2.VideoCapture(video_file)
	# get the information of the video file
	num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
	rate = video.get(cv2.CAP_PROP_FPS)

	count = 0

	while video.isOpened():
		ret, frame = video.read()
		boxes = []

		if not ret:
			print 'No frame read!'
			break

		if count == 0:
			PreviousFrame = frame.copy()
			# cv2.rectangle(FirstFrame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (255,255,0), 2)
			# target_pad, _, _, _ = cropPadImage(bbox, FirstFrame)
		else:
			for bbox in bboxes:
				target_pad, _, _, _ = cropPadImage(bbox, PreviousFrame)
				cur_search_region, search_location, edge_spacing_x, edge_spacing_y = cropPadImage(bbox, frame)

				bbox = track(cur_search_region, target_pad)
				bbox = BoundingBox(bbox[0, 0], bbox[0, 1], bbox[0, 2], bbox[0, 3])

				bbox.unscale(cur_search_region)
				bbox.uncenter(frame, search_location, edge_spacing_x, edge_spacing_y)

				boxes.append(bbox)


			PreviousFrame = frame.copy()
			frame = draw_fancybbox(frame, boxes)

		count += 1
		
		cv2.imshow('Result',frame)
		# cv2.waitKey()
		if cv2.waitKey(25) == 27:
			break


def track(search, target):

	net.blobs['image'].data.reshape(1, channels, height, width)
	net.blobs['target'].data.reshape(1, channels, height, width)
	net.blobs['bbox'].data.reshape(1, 4, 1, 1)

	search_region = preprocess(search)
	target_region = preprocess(target)

	net.blobs['image'].data[...] = search_region
	net.blobs['target'].data[...] = target_region
	net.forward()

	bbox_estimate = net.blobs['fc8'].data

	return bbox_estimate

def preprocess(image):
	num_channels = channels
	if num_channels == 1 and image.shape[2] == 3:
		image_out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	elif num_channels == 1 and image.shape[2] == 4:
		image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
	elif num_channels == 3 and image.shape[2] == 4:
		image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
	elif num_channels == 3 and image.shape[2] == 1:
		image_out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	else:
		image_out = image

	if image_out.shape != (height, width, channels):
		image_out = cv2.resize(image_out, (width, height), interpolation=cv2.INTER_CUBIC)

	image_out = np.float32(image_out)
	image_out -= np.array(mean_value)
	image_out = np.transpose(image_out, [2,0,1])

	return image_out

def cropPadImage(bbox_tight, image):
	"""TODO: Docstring for cropPadImage.
	:returns: TODO
	"""
	pad_image_location = computeCropPadImageLocation(bbox_tight, image)
	roi_left = min(pad_image_location.x1, (image.shape[1] - 1))
	roi_bottom = min(pad_image_location.y1, (image.shape[0] - 1))
	roi_width = min(image.shape[1], max(1.0, math.ceil(pad_image_location.x2 - pad_image_location.x1)))
	roi_height = min(image.shape[0], max(1.0, math.ceil(pad_image_location.y2 - pad_image_location.y1)))

	err = 0.000000001 # To take care of floating point arithmetic errors
	cropped_image = image[int(roi_bottom + err):int(roi_bottom + roi_height), int(roi_left + err):int(roi_left + roi_width)]
	output_width = max(math.ceil(bbox_tight.compute_output_width()), roi_width)
	output_height = max(math.ceil(bbox_tight.compute_output_height()), roi_height)
	if image.ndim > 2:
		output_image = np.zeros((int(output_height), int(output_width), image.shape[2]), dtype=image.dtype)
	else:
		output_image = np.zeros((int(output_height), int(output_width)), dtype=image.dtype)

	edge_spacing_x = min(bbox_tight.edge_spacing_x(), (image.shape[1] - 1))
	edge_spacing_y = min(bbox_tight.edge_spacing_y(), (image.shape[0] - 1))
	
	# if output_image[int(edge_spacing_y):int(edge_spacing_y) + cropped_image.shape[0], int(edge_spacing_x):int(edge_spacing_x) + cropped_image.shape[1]].shape != cropped_image.shape :
		# import pdb
		# pdb.set_trace()
		# print('debug')

	# rounding should be done to match the width and height
	output_image[int(edge_spacing_y):int(edge_spacing_y) + cropped_image.shape[0], int(edge_spacing_x):int(edge_spacing_x) + cropped_image.shape[1]] = cropped_image
	return output_image, pad_image_location, edge_spacing_x, edge_spacing_y


def computeCropPadImageLocation(bbox_tight, image):
	"""TODO: Docstring for computeCropPadImageLocation.
	:returns: TODO
	"""

	# Center of the bounding box
	bbox_center_x = bbox_tight.get_center_x()
	bbox_center_y = bbox_tight.get_center_y()

	image_height = image.shape[0]
	image_width = image.shape[1]

	# Padded output width and height
	output_width = bbox_tight.compute_output_width()
	output_height = bbox_tight.compute_output_height()

	roi_left = max(0.0, bbox_center_x - (output_width / 2.))
	roi_bottom = max(0.0, bbox_center_y - (output_height / 2.))

	# Padded roi width
	left_half = min(output_width / 2., bbox_center_x)
	right_half = min(output_width / 2., image_width - bbox_center_x)
	roi_width = max(1.0, left_half + right_half)

	# Padded roi height
	top_half = min(output_height / 2., bbox_center_y)
	bottom_half = min(output_height / 2., image_height - bbox_center_y)
	roi_height = max(1.0, top_half + bottom_half)

	# Padded image location in the original image
	objPadImageLocation = BoundingBox(roi_left, roi_bottom, roi_left + roi_width, roi_bottom + roi_height)
   
	return objPadImageLocation

def draw_fancybbox(im, boxes, max_obj=100, alpha=0.4):
	for bbox in boxes:
		cv2.rectangle(im, (int(bbox.x1),int(bbox.y1)), (int(bbox.x2),int(bbox.y2)), (136, 23, 251), 2)
		vis = Image.fromarray(im)

		mask = Image.fromarray(im.copy())
		draw = ImageDraw.Draw(mask)
		draw.rectangle((int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y1) + 24), fill=(171, 27, 253))
		draw.text((int(bbox.x1 + 5), int(bbox.y1) + 2),
				  'target', fill=(255, 255, 255), font=FONT20)
				
		im = np.array(Image.blend(vis, mask, alpha))

	return im

if __name__ == '__main__':
	file = '/Users/yangmin/PycharmProjects/learnopencv/tracking/videos/chaplin.mp4'
	bboxes = [BoundingBox(287, 23, 373, 343)]
	main(file, bboxes)


