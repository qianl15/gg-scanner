# This file is used to run Scanner experiments
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
from docopt import docopt

##########################################################################################################
# The kernel assumes DNN model is in PATH_TO_GRAPH with filename 'inception_v3_2016_08_28_frozen.pb'     #
# Example model can be downloaded from:                                                                  #
# https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz #
##########################################################################################################

script_dir = os.path.dirname(os.path.abspath(__file__))
PATH_TO_DATA = '/tmp/data'
PATH_TO_GRAPH = os.path.join(PATH_TO_DATA, 'inception_v3_2016_08_28_frozen.pb')

# What model to download.
MODEL_NAME = 'inception_v3_2016_08_28_frozen.pb'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'https://storage.googleapis.com/download.tensorflow.org/models/'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(PATH_TO_DATA, 'imagenet_slim_labels.txt')

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

doc = r"""
Usage: ./main.py [-n <num-instances>] [-b <decode-batch>] [-p] [<input>]

    -h,--help                               show this
    -n,--num-instances <num-instances>      number of pipeline instances (default: 1)
    -b,--decode-batch <decode-batch>        decode batch size (default: 75)
    -p,--show-progress                      show progress bar
"""
if __name__ == '__main__':
    options = docopt(doc)

    pipeline_instances = 1
    decode_batch = 75
    show_progress = False

    if not options['<input>']:
        print("Please provide one input video file!")
        exit(1)
    if options['--num-instances']:
        pipeline_instances = int(options['--num-instances'])
    if options['--decode-batch']:
        decode_batch = int(options['--decode-batch'])
    if options['--show-progress']:
        show_progress = True
    print('Run with pipeline_instances={:d}, decode_batch={:d}'.format(pipeline_instances, decode_batch))

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
                tar_file.extract(f, PATH_TO_DATA)
            if 'imagenet_slim_labels.txt' in file_name:
                tar_file.extract(f, PATH_TO_DATA)
        print("Successfully downloaded DNN Model.")

    movie_path = options['<input>']
    print('Detecting objects in movie {}'.format(movie_path))
    movie_name = os.path.splitext(os.path.basename(movie_path))[0]

    bundled_data_list = []
    sample_stride = 1

    # Preparation: register python kernel
    db = Database()
    db.register_op('ImgLabel',
                   [('frame', ColumnType.Video)],
                   ['bundled_data'])
    kernel_path = script_dir + '/label_image_kernel.py'
    db.register_python_kernel('ImgLabel', DeviceType.CPU, kernel_path)

    # Step 0: ingest video
    start = now()
    [input_table], failed = db.ingest_videos([('ggExp', movie_path)],
                                             force=True)
    stop = now()
    delta = stop - start
    print('Time to ingest: {:.4f}s, {} frames'.format(delta, input_table.num_rows()))

    # Step 1: construct the graph and do analysis on frames
    frame = db.sources.FrameColumn()
    strided_frame = frame.sample()

    # Call the newly created object detect op
    objdet_frame = db.ops.ImgLabel(frame = strided_frame)

    output_op = db.sinks.Column(columns={'bundled_data': objdet_frame})
    job = Job(
        op_args={
            frame: db.table('ggExp').column('frame'),
            strided_frame: db.sampler.strided(sample_stride),
            output_op: 'example_obj_detect',
        }
    )
    [out_table] = db.run(output=output_op, jobs=[job], force=True,
                         pipeline_instances_per_node=pipeline_instances,
                         work_packet_size=decode_batch,
                         show_progress=show_progress)
    stop2 = now()
    delta = stop2 - stop
    print('Time to analysis: {:.4f}s'.format(delta))

    # Step 2: extract output and write to files
    # bundled_data_list is a list of bundled_data
    # bundled data format: [top 5 pair of [class, probability] ]
    bundled_data_list = [pickle.loads(top5)
                         for top5 in tqdm(
                                 out_table.column('bundled_data').load())]

    # Print out results to files, one output file for one frame
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
    delta = stop3 - stop2
    print('Time to write output: {:.4f}s'.format(delta))
    print('Total end-end time: {:.4f}s'.format(stop3 - start))
    print('Total time exclude ingest: {:.4f}s'.format(stop3 - stop))
    print('Successfully completed {:s}.mp4 \n'.format(movie_name))

    exit(0)
