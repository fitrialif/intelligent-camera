###############################################################################
#
# \file    Engine.py
# \author  Sudnya Diamos <sudnyadiamos@gmail.com>
# \date    Saturday August 12, 2017
# \brief   Engine class that drives the classification pipeline
#
###############################################################################

import argparse
import logging

import tensorflow.contrib.slim as slim

from engine.InputDataProducer import InputDataProducer
from engine.ResultProcessor import ResultProcessor
from engine.ModelFactory import ModelFactory


logger = logging.getLogger("Engine")


class Engine:
    def __init__(self):
        #create the input and output placeholders
        self.logits, self.inputData = ModelFactory.create()

        imageWidth  = self.logits.size()[2]
        imageHeight = self.logits.size()[1]

        #create an instance of the dataproducer
        self.dataProducer = InputDataProducer(imageWidth, imageHeight)

        #create an instance of the result processor
        self.resultProcessor = ResultProcessor()



    def run(self, inputImages):
        #session.run(inputImages) on logits -- executes the graph
        inputData = self.dataProducer.process(inputImages)

        logits = None
        with tf.Session() as session:
            logits = session.run([self.logits],
                feed_dict={self.inputData : inputData})

        #pass the logits to the results processor instance
        resultClasses = self.resultProcessor.getLabels(logits)

        #return list of inputImages and output of the results processor

        return inputImages, resultClasses



