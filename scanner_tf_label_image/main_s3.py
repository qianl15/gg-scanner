# Start virtualenv for tensorflow!
TF_PATH = "/opt/tensorflow_env/bin/activate_this.py"

execfile(TF_PATH, dict(__file__=TF_PATH))

import tensorflow as tf
from scannerpy import Database, Job, ColumnType, DeviceType
import os
import sys
import math
from tqdm import tqdm
import six.moves.urllib as urllib
import tarfile
import pickle

from timeit import default_timer as now

##################################################################################################
# The kernel assumes DNN model is in PATH_TO_GRAPH with filename 'inception_v3_2016_08_28_frozen.pb'     #
# Example model can be downloaded from:                                                          #
# https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz #
##################################################################################################

script_dir = os.path.dirname(os.path.abspath(__file__))
PATH_TO_REPO = script_dir
PATH_TO_GRAPH = os.path.join(PATH_TO_REPO, 'data', 'inception_v3_2016_08_28_frozen.pb')

# What model to download.
MODEL_NAME = 'inception_v3_2016_08_28_frozen.pb'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(PATH_TO_REPO, 'data', 'imagenet_slim_labels.txt')

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

if __name__ == '__main__':
#    if len(sys.argv) <= 1:
#        print('Usage: {:s} path/to/your/video/file.mp4'.format(sys.argv[0]))
#        sys.exit(1)

#    # Download the DNN model if not found in PATH_TO_GRAPH
#    if not os.path.isfile(PATH_TO_GRAPH):
#        print("DNN Model not found, now downloading...")
#        opener = urllib.request.URLopener()
#        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#        tar_file = tarfile.open(MODEL_FILE)
#        for f in tar_file.getmembers():
#            file_name = os.path.basename(f.name)
#            if 'frozen_inference_graph.pb' in file_name:
#                tar_file.extract(f, PATH_TO_REPO)
#                break
#        print("Successfully downloaded DNN Model.")
#
#    movie_path = sys.argv[1]
    if len(sys.argv) <= 1:
        print('Usage: {:s} videos/<file.mp4>'.format(sys.argv[0]))
        sys.exit(1)

    movie_path = 'videos/' + sys.argv[1]

    #movie_path = "videos/4kvid_chunk6kf.mp4"
    print('Detecting objects in movie {}'.format(movie_path))
    movie_name = os.path.splitext(os.path.basename(movie_path))[0]

    bundled_data_list = []
    sample_stride = 30

    labels = load_labels(PATH_TO_LABELS) # load labels
    start = now()

    workerList = ['ip-172-31-7-237:5002', 'ip-172-31-15-139:5002',
                  'ip-172-31-11-111:5002', 'ip-172-31-14-230:5002']
    with Database(config_path="/home/ubuntu/.scanner_s3.toml", workers=workerList) as db:
    #with Database(config_path="/home/ubuntu/.scanner_s3.toml") as db:
        [input_table], failed = db.ingest_videos([('example', movie_path)],
                                                 force=True)
        stop = now()
        delta = stop - start
        print('Time to ingest: {:.4f}s, {} frames'.format(delta, input_table.num_rows()))

        db.register_op('ImgLabel',
                       [('frame', ColumnType.Video)],
                       ['bundled_data'])
        kernel_path = script_dir + '/label_image_kernel.py'
        db.register_python_kernel('ImgLabel', DeviceType.CPU, kernel_path)
        frame = db.sources.FrameColumn()
        strided_frame = frame.sample()

        # Call the newly created object detect op
        objdet_frame = db.ops.ImgLabel(frame = strided_frame)

        output_op = db.sinks.Column(columns={'bundled_data': objdet_frame})
        job = Job(
            op_args={
                frame: db.table('example').column('frame'),
                strided_frame: db.sampler.strided(sample_stride),
                output_op: 'example_obj_detect',
            }
        )
        [out_table] = db.run(output=output_op, jobs=[job], force=True,
                             pipeline_instances_per_node=1,
                             work_packet_size=25)

        stop2 = now()
        delta = stop2 - stop
        print('Time to analysis: {:.4f}s'.format(delta))
        print('Extracting data from Scanner output...')

        # bundled_data_list is a list of bundled_data
        # bundled data format: [top 5 pair of [class, probability] ]
        bundled_data_list = [pickle.loads(top5)
                             for top5 in tqdm(
                                     out_table.column('bundled_data').load())]
    #  Print out results to files, one output file for one frame
    labels = load_labels(PATH_TO_LABELS) # load labels
    count = 0
    for row in bundled_data_list:
        fileName = 'frame{:06d}.out'.format(count)
        f = open(fileName, 'w')
        for pair in row:
            ind = int(pair[0])
            prob = pair[1]
            f.write('{} ({:d}): {:.7f}\n'.format(labels[ind], ind, prob))
        f.close()
        count += 1

    stop3 = now()
    print('Total end-end time: {:.4f}s'.format(stop3 - start))
    print('Successfully completed {:s}.mp4'.format(movie_name))
exit(0)
