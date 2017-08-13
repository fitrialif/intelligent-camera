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

logger = logging.getLogger("ResultProcessor")

class ResultProcessor:
    def __init__(self):
        pass

    def getLabels(self, pd):
        #TODO: [probabilities] -> pick max entry -> class label
        return [{"label": "cat"}]


