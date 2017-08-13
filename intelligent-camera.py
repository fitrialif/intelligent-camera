
import argparse

from driver.IntelligentCamera import IntelligentCamera

def main():
    parser = argparse.ArgumentParser(description='An application for adding intelligence to "
        "a collection of cameras.')

    parser.add_argument('-i', '--input', default='.',
                        help='The input directory to search for images in.')
    parser.add_argument('-o', '--output', default='./output',
                        help='The output directory to store processed images in.')

    intelligentCamera = IntelligentCamera(vars(parser.parse_args()))

    intelligentCamera.run()

