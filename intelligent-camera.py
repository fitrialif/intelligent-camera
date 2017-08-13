
import argparse
import logging

from driver.IntelligentCamera import IntelligentCamera

def main():
    parser = argparse.ArgumentParser(description='An application for adding intelligence to '
         + 'a collection of cameras.')

    parser.add_argument('-i', '--input', default='.',
                        help='The input directory to search for images in.')
    parser.add_argument('-o', '--output', default='./output',
                        help='The output directory to store processed images in.')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='Turn on debugging messages.')

    options = vars(parser.parse_args())

    if options['verbose']:
        logging.basicConfig(level=logging.DEBUG)

    intelligentCamera = IntelligentCamera(options)

    intelligentCamera.run()

if __name__ == "__main__":
    main()

