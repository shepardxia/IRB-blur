import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json as js
import PIL.Image as Image
import time


def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

model_path = './weights/weights_plate_v1.0.0.pb'
graph = load_pb(model_path)
sess = tf.function(graph)



image_tensor = graph.get_tensor_by_name('image_tensor: 0')
num_det = graph.get_tensor_by_name('num_detections: 0')
det_scores = graph.get_tensor_by_name('detection_scores: 0')
det_boxes = graph.get_tensor_by_name('detection_boxes: 0')

image = cv2.imread('./images/10.jpg')
np_images = np.array([image])
num_boxes, scores, boxes = sess.run([num_det, det_scores, det_boxes], feed_dict={image_tensor: np_images})


#before = time.time()
#print(before)

#for i in range(300):
#    image = cv2.imread('./images/10.jpg')
#    np_images = np.array([image])
#    num_boxes, scores, boxes = sess.run([num_det, det_scores, det_boxes], feed_dict={image_tensor: np_images})

#after = time.time()
#print(after)

#print("total time used:", (after - before))

#rets = []
#
#for i in range(len(scores[0])):
#    if scores[0][i] > 0.01:
#        rets.append(boxes[0][i].tolist())


#ret_img = tf.image.draw_bounding_boxes(np_images, [rets], [[255, 0, 0]])
#print(ret_img)
#plt.imshow(tf.cast(ret_img[0], tf.int32))
#plt.show()

#cv2.imshow('hey', ret_img[0].numpy())
#cv2.waitKey(0)


