###############################################################################
#
# \file    InputDataProducer.py
# \author  Sudnya Diamos <sudnyadiamos@gmail.com>
# \date    Saturday August 12, 2017
# \brief   Class that converts the given input image (variable resolutions) to
#          a tensor
#
###############################################################################

import argparse
import logging

from PIL import Image

logger = logging.getLogger("InputDataProducer")

def getData(image, xSize, ySize):
    pixel = image.load()
    data = np.array((1, ySize, xSize, 3))
    for x in range(xSize):
        for y in range(ySize):
            (red,green,blue) = pixel[x,y]
            data[0, y, x, 0] = red
            data[0, y, x, 1] = green
            data[0, y, x, 2] = blue

    return data

class InputDataProducer:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def process(self, inputImages):
        outputData = []
        for inputImage in inputImages:
            resizedImage = inputImage.resize(self.x, self.y)
            outputData.append(getData(resizedImage, x, y))

        return np.concatenate(outputData)

