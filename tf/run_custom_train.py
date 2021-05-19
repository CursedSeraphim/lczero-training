import custom_train
import argparse
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    argparser = argparse.ArgumentParser(description=\
    'Tensorflow pipeline for training Leela Chess.')
    argparser.add_argument('--cfg',
                            type=argparse.FileType('r'),
                            help='yaml configuration with training parameters')
    argparser.add_argument('--output',
                            type=str,
                            help='file to store weights in')
    cmd = argparser.parse_args()
    cmd.cfg = open('128x10-t60-2.yaml')
    cmd.net = '128x10-t60-2-5300.pb.gz'

    custom_train.main(cmd)