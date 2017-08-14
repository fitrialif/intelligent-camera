

from engine.Engine import Engine

from PIL import Image

import imghdr

import os
import logging

def isValidImage(path):
    if imghdr.what(path) == None:
        return False

    return True

logger = logging.getLogger("IntelligentCamera")

class IntelligentCamera:
    def __init__(self, options):
        self.options = options
        self.engine = Engine()

    def run(self):
        images = self.findImages()

        self.classifyImages(images)

    def findImages(self):
        images = []
        for directoryName, subdirectoryList, fileList in os.walk(self.options['input']):
            for fileName in fileList:
                path = os.path.join(directoryName, fileName)
                if isValidImage(path):
                    logger.info("Found image to process '" + path + "'")
                    images.append(path)

        return images

    def classifyImages(self, imagePaths):
        images = [Image.open(imagePath) for imagePath in imagePaths]

        results = self.engine.run(images)

        index = 0
        for resultImage, resultLabel in results:
            label = resultLabel['label']
            self.saveImage(resultImage, label, imagePaths[index])
            index += 1

    def saveImage(self, resultImage, label, inputImagePath):

        fileName = os.path.split(inputImagePath)[1]
        outputPath = os.path.join(self.options['output'], label, fileName)

        if not os.path.exists(os.path.dirname(outputPath)):
            os.makedirs(os.path.dirname(outputPath))
        
        resultImage.save(outputPath)


