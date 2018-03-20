import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import time
import glob

from io import StringIO
from PIL import Image

import matplotlib.pyplot as plt

from utils import visualization_utils as vis_util
from utils import label_map_util

from multiprocessing.dummy import Pool as ThreadPool



############## more ##############
import yaml
import cv2
from stuff.helper import FPS2, WebcamVideoStream
from tensorflow.core.framework import graph_pb2

## OG PARAMETERS ##
MAX_NUMBER_OF_BOXES = 10
MINIMUM_CONFIDENCE = 0.9

PATH_TO_LABELS = 'annotations/label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'test_images'

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=sys.maxsize, use_display_name=True)
CATEGORY_INDEX = label_map_util.create_category_index(categories)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'output_inference_graph'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'



## LOAD CONFIG PARAMS ##
# with open("config.yml", 'r') as ymlfile:
#     cfg = yaml.load(ymlfile)

# video_input         = cfg['video_input']          # Input Must be OpenCV readable 
# visualize           = cfg['visualize']            # Disable for performance increase
# vis_text            = cfg['vis_text']             # Display fps on visualization stream
# max_frames          = cfg['max_frames']           # only used if visualize==False
# width               = cfg['width']                # 300x300 is used by SSD_Mobilenet -> highest fps
# height              = cfg['height']
# fps_interval        = cfg['fps_interval']         # Intervall [s] to print fps in console
# allow_memory_growth = cfg['allow_memory_growth']  # limits memory allocation to the actual needs
# det_interval        = cfg['det_interval']         # intervall [frames] to print detections to console
# det_th              = cfg['det_th']               # detection threshold for det_intervall
# model_name          = cfg['model_name']
# model_path          = cfg['model_path']
# label_path          = cfg['label_path']
# num_classes         = cfg['num_classes']
# split_model         = cfg['split_model']          # Splits Model into a GPU and CPU session (currently only works for ssd_mobilenets)
# log_device          = cfg['log_device']           # Logs GPU / CPU device placement
# ssd_shape           = cfg['ssd_shape']

###### Manual loading cause we don't have a config file yet :| ########
model_name = 'ssd_mobilenet_v1_coco'
video_input = 0              # Input Must be OpenCV readable 
visualize = True             # Disable for performance increase
max_frames = 500             # only used if visualize==False
width = 300                  # 300x300 is used by SSD_Mobilenet -> highest fps
height = 300
fps_interval = 3             # Intervall [s] to print fps in console
bbox_thickness = 8           # thickness of bounding boxes printed to the output image
allow_memory_growth = True   # restart python to apply changes on memory usage
det_intervall = 75           # intervall [frames] to print detections to console
det_th = 0.5                 # detection threshold for det_intervall

#####################################################


def detect_objects(image_path):
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        CATEGORY_INDEX,
        min_score_thresh=MINIMUM_CONFIDENCE,
        use_normalized_coordinates=True,
        line_thickness=8)
    fig = plt.figure()
    fig.set_size_inches(16, 9)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(image_np, aspect = 'auto')
    plt.savefig('output/{}'.format(image_path), dpi = 62)
    plt.close(fig)

def object_detection(video_input,visualize,max_frames,width,height,fps_interval,bbox_thickness, \
                     allow_memory_growth,det_intervall,det_th,model_name):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=allow_memory_growth
        

    cur_frames = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config = config) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # fps calculation
            fps = FPS2(fps_interval).start()
            # Start Video Stream
            video_stream = WebcamVideoStream(video_input,width,height).start()
            print ("Press 'q' to Exit")
            while video_stream.isActive():
                image_np = video_stream.read()
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
                if visualize:
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        CATEGORY_INDEX,
                        min_score_thresh=MINIMUM_CONFIDENCE,
                        use_normalized_coordinates=True,
                        line_thickness=bbox_thickness)
                    cv2.imshow('object_detection', image_np)
                    # print(boxes)
                    # print bounding corners of boxes when confidence is > minimum confidence (the ones you're drawing boxes around)
                    print("NEW FRAME")
                    for i, box in enumerate(np.squeeze(boxes)):
                        if(np.squeeze(scores)[i] > MINIMUM_CONFIDENCE):
                            # This uses actual coordinates based on size of image - remove height and width to use normalized coordinates
                            ymin = box[0]*height
                            xmin = box[1]*width
                            ymax = box[2]*height
                            xmax = box[3]*width
                            print ('Top left')
                            print ("(" + str(xmin) + "," + str(ymin) + ")")
                            print ('Bottom right')
                            print ("(" + str(xmax) + "," + str(ymax) + ")")

                    print()
                    # Exit Option--BROKEN
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    cur_frames += 1
                    for box, score, _class in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):
                        if cur_frames%det_intervall==0 and score > det_th:
                            label = category_index[_class]['name']
                            print(label, score, box)
                    if cur_frames >= max_frames:
                        break
                # fps calculation
                fps.update()
    
    # End everything
    fps.stop()
    video_stream.stop()     
    cv2.destroyAllWindows()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))


# Load model into memory
print('Loading model...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

print('detecting...')
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')    

    object_detection(video_input, visualize, max_frames, width, height, fps_interval, bbox_thickness, allow_memory_growth, det_intervall, det_th, model_name)
