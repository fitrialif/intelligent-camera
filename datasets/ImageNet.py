
from datasets.ImageNetSynsets import ImageNetSynsets

class ImageNet:
    def __init__(self):
        self.usedLabels = ImageNetSynsets.getUsedLabels()

    def getLabelForLogit(self, logitId):
        return self.usedLabels[logitId]



