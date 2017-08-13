###############################################################################
#
# \file    ResultProcessor.py
# \author  Sudnya Diamos <sudnyadiamos@gmail.com>
# \date    Saturday August 12, 2017
# \brief   Class that converts a probability distribution over classes to class
#          label and returns result as a json object
###############################################################################

import argparse
import logging
import numpy

logger = logging.getLogger("ResultProcessor")

class ResultProcessor:
    def __init__(self):
        pass

    def getLabels(self, pd, labelMapper):
        #TODO: [probabilities] {batch, probs} -> pick max entry -> class label
        batchSize  = pd.shape[0]
        labelCount = pd.shape[1]

        labels = []
        for batchElement in range(batchSize):
            probs = numpy.reshape(pd[batchElement:batchElement + 1, :], (labelCount))
            mostLikelyLabelIndex = numpy.argmax(probs)
            label = labelMapper.getLabelForLogit(mostLikelyLabelIndex)

            labels.append({"label" : label})

        return labels

