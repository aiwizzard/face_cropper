"""
Script to crop faces from image or video
Usage:
 Crop faces from image:
    python crop_faces.py --type image --image image.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel --output output_path

 Crop faces from video:
    python crop_faces.py --type video --video video.mp4 --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel --output output_path
"""

import argparse
import cv2
import numpy as np
import secrets
import os

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', required=True,
                help='type of the file')
parser.add_argument('-i', '--image',
                help='path to the image file')
parser.add_argument('-v', '--video',
                help='path to the video file')
parser.add_argument('-p', '--prototxt', required=True,
                help='path to the prototxt file')
parser.add_argument('-m', '--model', required=True,
                help='path to the pretrained model')
parser.add_argument('-c', '--confidence', type=float, default=0.5,
                help='minimum probability to filter the weak detections')
parser.add_argument('-o', '--output', required=True,
                help='path to save the cropped faces')
args = vars(parser.parse_args())

# load the serialized model from the disk
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

# this will detect the faces and crop them
def detect(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args['confidence']:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            face = image[startY:endY, startX:endX] # cropping the image
            save(face)

# this will save the images to the user specified directory
def save(image):
    random_name = secrets.token_hex(8)
    image_name = random_name + '.jpg'
    image_path = os.path.join(args['output'], image_name)
    cv2.imwrite(image_path, image)


if args['type'] == 'image':
    if args['image'] is None:
        parser.error('You should specify an image')
    else:
        # load the image from disk
        image = cv2.imread(args['image'])
        detect(image)

if args['type'] == 'video':
    if args['video'] is None:
        parser.error('You should specify a video')
    else:
        video = cv2.VideoCapture(args['video'])
        while True:
            ret, frame = video.read()
            if not ret:
                break
            else:
                detect(frame)