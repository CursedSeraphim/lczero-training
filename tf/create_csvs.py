import os.path
import argparse
import custom_train
from multiprocessing import freeze_support

def run_custom_train(layer, path):
    freeze_support()
    argparser = argparse.ArgumentParser(description=\
    'Tensorflow pipeline for training Leela Chess.')
    argparser.add_argument('--cfg',
                            type=argparse.FileType('r'),
                            help='yaml configuration with training parameters')
    argparser.add_argument('--output',
                            type=str,
                            help='file to store weights in')
    argparser.add_argument('--layer',
                            type=str,
                            help='names of network layer the logs of which should be stored to the csv')    
    argparser.add_argument('--path',
                            type=str,
                            help='paths for storing the csv')    
    cmd = argparser.parse_args()
    cmd.layer = layer
    cmd.path = path
    cmd.cfg = open('128x10-t60-2.yaml')
    cmd.net = '128x10-t60-2-5300.pb.gz'

    custom_train.main(cmd)


layers = ['activation', 'activation_1', 'activation_2', 'activation_3']
# layers = ['moves_left/dense2', 'value/dense2', 'apply_policy_map_2', 'moves_left/dense1', 'value/dense1', 'policy', 'flatten_1', 'flatten', 'activation_31', 'activation_32', 'activation_33']
prefix = 'a0sf_'
paths = [(prefix+s+'.csv').replace('/','-') for s in layers]

run_custom_train(layers, paths)
