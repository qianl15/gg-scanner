# Mostly taken from: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image
import numpy as np
import tensorflow as tf
import cv2
import os
from scannerpy.stdlib import kernel
import pickle

import tarfile
import six.moves.urllib as urllib
from timeit import default_timer as now
##################################################################################################
# Assume that DNN model is located in PATH_TO_GRAPH with filename 'inception_v3_2016_08_28_frozen.pb'    #
# Example model can be downloaded from:                                                          #
# https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz #
##################################################################################################
# What model to download.
MODEL_NAME = 'inception_v3_2016_08_28_frozen.pb'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'https://storage.googleapis.com/download.tensorflow.org/models/'

PATH_TO_REPO = '/tmp'

PATH_TO_GRAPH = os.path.join(PATH_TO_REPO, 'data', 'inception_v3_2016_08_28_frozen.pb')

class ImgLabelKernel(kernel.TensorFlowKernel):
    def build_graph(self):
        dnn = tf.Graph()
        with dnn.as_default():
            od_graph_def = tf.GraphDef()

            # Download the DNN model if not found in PATH_TO_GRAPH
            if not os.path.isfile(PATH_TO_GRAPH):
                print("DNN Model not found, now downloading...")
                opener = urllib.request.URLopener()
                downloadFile = os.path.join('/tmp', MODEL_FILE)
                opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, downloadFile)
                tar_file = tarfile.open(downloadFile)
                for f in tar_file.getmembers():
                    file_name = os.path.basename(f.name)
                    if 'inception_v3_2016_08_28_frozen.pb' in file_name:
                        tar_file.extract(f, os.path.join(PATH_TO_REPO, 'data'))
                        break
                print("Successfully downloaded and extracted DNN Model.")

            with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return dnn

    # Evaluate labeling image DNN model on a frame
    # Return bounding box position, class and score
    def execute(self, cols):
        #start = now()
        image = cols[0]
        # Must resize the image to 299 x 299 x 3!
        image = np.expand_dims(image, axis=0)
        input_layer = "input:0"
        output_layer = "InceptionV3/Predictions/Reshape_1:0"
        image_tensor = self.graph.get_tensor_by_name(input_layer)
        output_tensor = self.graph.get_tensor_by_name(output_layer)
        resized = tf.image.resize_bilinear(image, [299, 299])
        normalized = tf.divide(tf.subtract(resized, [0]), [255])
        sess = tf.Session()
        with sess.as_default():
            result_image = sess.run(normalized)
        sess.close()

        with self.graph.as_default():
            classes = self.sess.run(
                output_tensor,
                feed_dict={image_tensor: result_image})
            classes = np.squeeze(classes)
            topk = classes.argsort()[-5:][::-1]
            bundled_data = []
            for i in topk:
                bundled_data.append([i, classes[i]])
            bundled_np_data = np.array(bundled_data)
            bundled_bytes_data = pickle.dumps(bundled_np_data)
            #stop = now()
            #print('execute time: {:.4f}s'.format(stop - start))
            return [bundled_bytes_data]

KERNEL = ImgLabelKernel
