import tensorflow as tf
import glob
import os
files =  glob.glob("/gpfs/qlong/home/tzzhang/nlp_test/bert/mimic_examples_for_complete_128/*.tfrecord") 

filesSize = len(files)
cnt = 0 

for filename in files:
    cnt = cnt + 1
    print('checking %d/%d %s' % (cnt, filesSize, filename))
    try:
        for example in tf.python_io.tf_record_iterator(filename): 

            tf_example = tf.train.Example.FromString(example) 
    except :
        print("removing %s" % filename)
        os.remove(filename)