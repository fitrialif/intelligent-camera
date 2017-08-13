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
        self.batchSize = 2
        self.logits, self.inputData = ModelFactory.create(self.batchSize)

        imageWidth  = self.logits.size()[2]
        imageHeight = self.logits.size()[1]

        #create an instance of the dataproducer
        self.dataProducer = InputDataProducer(imageWidth, imageHeight)

        #create an instance of the result processor
        self.resultProcessor = ResultProcessor()

    def run(self, inputImages):
        batchCount = (len(inputImages) + self.batchSize - 1) / self.batchSize

        resultClasses = []
        for batch in range(batchCount):
            batchBegin = batch * self.batchSize
            batchEnd   = min(len(inputImages), batchBegin + self.batchSize)
            imageBatch = inputImages[batchBegin:batchEnd]

            dynamicBatchSize = batchEnd - batchBegin

            while len(imageBatch) < self.batchSize:
                imageBatch.append(imageBatch[0])

            #session.run(inputImages) on logits -- executes the graph
            inputData = self.dataProducer.process(imageBatch)

            logits = None
            with tf.Session() as session:
                logits = session.run([self.logits],
                    feed_dict={self.inputData : inputData})

            #pass the logits to the results processor instance
            resultClasses += self.resultProcessor.getLabels(logits[0:dynamicBatchSize])

            #return list of inputImages and output of the results processor

        return zip(inputImages, resultClasses)



