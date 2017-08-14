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
import numpy

from PIL import Image

logger = logging.getLogger("InputDataProducer")

def getData(image, xSize, ySize):
    pixel = image.load()
    data = numpy.zeros((1, ySize, xSize, 3), dtype=numpy.uint8)
    for x in range(xSize):
        for y in range(ySize):
            (red,green,blue) = pixel[x,y]
            data[0, y, x, 0] = red
            data[0, y, x, 1] = green
            data[0, y, x, 2] = blue

    #Image.fromarray(numpy.asarray(numpy.reshape(data, (ySize, xSize, 3)))).show()
    return data

class InputDataProducer:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def process(self, inputImages):
        outputData = []
        for inputImage in inputImages:
            resizedImage = inputImage.resize((self.x, self.y))
            outputData.append(getData(resizedImage, self.x, self.y))

        return numpy.concatenate(outputData)

