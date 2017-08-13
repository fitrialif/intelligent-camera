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

import tensorflow as tf
from models.InceptionResnetV2 import inception_resnet_v2

logger = logging.getLogger("ModelFactory")

inceptionResnetV2Path = "inception_resnet_v2_2016_08_30.ckpt"

class ModelFactory:

    @staticmethod
    def create(batchSize):
        #call the inception_resnet_v2 function

        imageHeight = inception_resnet_v2.default_image_size
        imageWidth  = inception_resnet_v2.default_image_size

        inputData = tf.placeholder(tf.float32, [batchSize, imageHeight, imageWidth, 3])

        logits, endpts = inception_resnet_v2(inputData,
                                             num_classes=1,
                                             is_training=False,
                                             dropout_keep_prob=0.8,
                                             reuse=None,
                                             scope='InceptionResnetV2',
                                             create_aux_logits=True)

        #load the input placeholder with the ckpt file
        with tf.Session() as session:
            tf.saved_model.loader.load(session, ["InceptionResnetV2"],
                                       inceptionResnetV2Path)

        return logits, inputData



