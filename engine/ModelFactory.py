###############################################################################
#
# \file    ModelFactory.py
# \author  Sudnya Diamos <sudnyadiamos@gmail.com>
# \date    Saturday August 12, 2017
# \brief   Class that creates the inception_resnet_v2 model and loads weights
#          into the model
#
###############################################################################

import argparse
import logging
import numpy

import tensorflow as tf
from models.InceptionResnetV2 import inception_resnet_v2
from models.InceptionResnetV2 import inception_resnet_v2_arg_scope

from datasets.ImageNet import ImageNet 

logger = logging.getLogger("ModelFactory")

inceptionResnetV2Path = "inception_resnet_v2_2016_08_30.ckpt"

class ModelFactory:

    @staticmethod
    def create(batchSize):
        #call the inception_resnet_v2 function

        imageHeight = inception_resnet_v2.default_image_size
        imageWidth  = inception_resnet_v2.default_image_size

        inputData = tf.placeholder(tf.float32,
            [batchSize, imageHeight, imageWidth, 3])

        with tf.contrib.slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, endpts = inception_resnet_v2(inputData,
                                                 num_classes=1001,
                                                 is_training=False,
                                                 dropout_keep_prob=1.0,
                                                 reuse=None,
                                                 scope='InceptionResnetV2',
                                                 create_aux_logits=True)

        restorer = tf.train.Saver()

        #load the input placeholder with the ckpt file
        session = tf.Session()
        restorer.restore(session, inceptionResnetV2Path)
        #tf.saved_model.loader.load(session, ["InceptionResnetV2"],
        #                           inceptionResnetV2Path)
        #variablesToRestore = tf.contrib.slim.get_variables_to_restore()
        #l = session.run([logits],
        #    feed_dict={inputData : numpy.zeros((batchSize, imageHeight, imageWidth, 3))})

        return logits, inputData, session, ImageNet()



