from engine.Engine import Engine
from PIL import Image

import json

class EngineTester(unittest.TestCase):
    def setUp(self):
        self.engine = Engine()

class TestEmptyImageList(EngineTester):
    def runTest(self):
        results = self.engine.run([])

        assert len(results) == 0

def randomImage(dims):
    testImage = Image.new("RGB", dims, (255,255,255))
    pixel = testImage.load()
    for x in range(dims[0]):
        for y in range(dims[1]):
            red = random.randrange(0,255)
            blue = random.randrange(0,255)
            green = random.randrange(0,255)
            pixel[x,y]=(red,blue,green)
    return testImage

class TestRandomImage(EngineTester):
    def runTest(self):
        results = self.engine.run([randomImage((64, 64))])
        assert len(results > 0)

        for image, label in results:
            assert isImage(image)
            assert 'className' in label
            labelJson = json.dumps(label)

