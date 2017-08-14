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
            logger.info("Most like label: " + str(mostLikelyLabelIndex) \
                    + " with score: " + str(probs[mostLikelyLabelIndex]))
            label = labelMapper.getLabelForLogit(mostLikelyLabelIndex)

            top5LabelIndices = numpy.argpartition(probs, -5)[-5:]

            top5LabelIndices = reversed(top5LabelIndices[numpy.argsort(probs[top5LabelIndices])])

            top5Labels = [labelMapper.getLabelForLogit(index) for index in top5LabelIndices]

            result = {"label" : label, "top-5-labels" : top5Labels}
            
            logger.info(" result: " + str(result))

            labels.append(result)

        return labels

