import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
from scipy.stats import norm
import os
import time
import cv2

plt.style.use('ggplot')

# Colors (one for each class)
cmap = ImageColor.colormap
print("Number of colors =", len(cmap))
COLOR_LIST = sorted([c for c in cmap.keys()])

# Utility funcs
def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)
    
    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes

def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].
    
    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width
    
    return box_coords

def draw_boxes(image, boxes, classes, thickness=4):
    """Draw bounding boxes on the image"""
    # draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
        color = COLOR_LIST[class_id]
        cv2.rectangle(image, (left, top), (right, bot), (255,0,0), thickness)
        #draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)
        
def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


# Frozen inference graph files. NOTE: change the path to where you saved the models.
#PATH_TO_FROZEN_GRAPH = '/home/jose/GitHubs/model-training/models/ssd_inception_v2_coco_2017_11_17/'
PATH_TO_FROZEN_GRAPH = '/home/jose/GitHubs/model-training/models/new_graph_10/'
FROZEN_GRAPH = PATH_TO_FROZEN_GRAPH + 'frozen_inference_graph.pb'
detection_graph = load_graph(FROZEN_GRAPH)

# The input placeholder for the image.
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

LIGHTS = ['Green', 'Yellow', 'Red', 'Unknown']

def classify_image(image, detection_graph, loop=False):
#    image_np = np.asarray(image, dtype=np.uint8)
#    image_np = np.dstack((image_np[:, :, 2], image_np[:, :, 1], image_np[:, :, 0]))

    with tf.Session(graph=detection_graph) as sess:                
        # Actual detection.
        (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes], 
                                            feed_dict={image_tensor: np.expand_dims(image, 0)})
        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
    
        confidence_cutoff = 0.1
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

        if len(scores)>0:
            this_class = int(classes[np.argmax(scores)])
        else:
            this_class = 4
        
        if not loop:
            print(classes)
            print(scores)
            print("Class: ", LIGHTS[this_class-1])
            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            # Each class with be represented by a differently colored box
#            print(image.size)
            width, height = image.shape[1], image.shape[0]
            box_coords = to_image_coords(boxes, height, width)      
            draw_boxes(image, box_coords, classes)
            plt.figure(figsize=(12, 8))
            plt.grid(False)
            plt.imshow(image) 
        return LIGHTS[this_class-1]


def loop_over(TEST_IMAGES):
    win = 0
    total = 0
    t0 = time.time()
    for image_file in TEST_IMAGES:
        image = cv2.imread(PATH_TO_TEST_IMAGES_DIR + '/' + image_file)
        infer_light = classify_image(image, detection_graph, loop= True)
        correct_light = image_file.split('_')[0]
        total += 1
        if infer_light==correct_light: win += 1
        print("Class / Infer: %s / %s (%i%%)" % (correct_light, infer_light, 100*(win/total)))
    print("%.2f seconds" % (time.time() - t0))
        
    
PATH_TO_TEST_IMAGES_DIR = 'imgs_40'
TEST_IMAGES = [f for f in os.listdir(PATH_TO_TEST_IMAGES_DIR) if os.path.isfile(os.path.join(PATH_TO_TEST_IMAGES_DIR, f))]
loop_over(TEST_IMAGES)

#single_path = 'imgs_40'
#single_file = 'Red_0.0_546.png'
##single_image = Image.open(single_path + '/' + single_file)
#single_image = cv2.imread(single_path + '/' + single_file)
#classify_image(single_image, detection_graph, loop= False)






    