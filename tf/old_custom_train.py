#!/usr/bin/env python3
import argparse
import os
import yaml
from tfprocess import TFProcess
from net import Net
from chunkparser import ChunkParser
import multiprocessing as mp

cfg_path = "128x10-t60-2.yaml"
net_path = "128x10-t60-2-5300.pb.gz"
ignore_errors = False

with open(cfg_path,'r') as f:
  cfg = yaml.safe_load(f.read())



print(yaml.dump(cfg, default_flow_style=False))
# START_FROM = args.start

tfp = TFProcess(cfg)
tfp.init_net_v2()
tfp.replace_weights_v2(net_path, ignore_errors)
# tfp.global_step.assign(START_FROM)

# root_dir = os.path.join(cfg['training']['path'], cfg['name'])
# if not os.path.exists(root_dir):
#     os.makedirs(root_dir)
# # tfp.manager.save(checkpoint_number=START_FROM)
# print("Wrote model to {}".format(tfp.manager.latest_checkpoint))

cfg['dataset']['input_train'] = "tf/data/*/"
cfg['dataset']['input_test'] = "tf/data/*/"

import glob
for d in glob.glob(cfg['dataset']['input_train']):
  print(d)

from train import get_latest_chunks, get_all_chunks, get_chunks

print(yaml.dump(cfg, default_flow_style=False))

num_chunks = cfg['dataset']['num_chunks']
allow_less = cfg['dataset'].get('allow_less_chunks', False)
train_ratio = cfg['dataset']['train_ratio']
experimental_parser = cfg['dataset'].get('experimental_v5_only_dataset',
                                          False)
num_train = int(num_chunks * train_ratio)
num_test = num_chunks - num_train
sort_type = cfg['dataset'].get('sort_type', 'mtime')
if sort_type == 'mtime':
    sort_key_fn = os.path.getmtime
elif sort_type == 'number':
    sort_key_fn = game_number_for_name
elif sort_type == 'name':
    sort_key_fn = identity_function
else:
    raise ValueError('Unknown dataset sort_type: {}'.format(sort_type))
if 'input_test' in cfg['dataset']:
    train_chunks = get_latest_chunks(cfg['dataset']['input_train'],
                                      1, allow_less, sort_key_fn)
    test_chunks = get_latest_chunks(cfg['dataset']['input_test'], 1,
                                    allow_less, sort_key_fn)
else:
    chunks = get_latest_chunks(cfg['dataset']['input'], num_chunks,
                                allow_less, sort_key_fn)
    if allow_less:
        num_train = int(len(chunks) * train_ratio)
        num_test = len(chunks) - num_train
    train_chunks = chunks[:num_train]
    test_chunks = chunks[num_train:]

shuffle_size = cfg['training']['shuffle_size']
total_batch_size = cfg['training']['batch_size']
batch_splits = cfg['training'].get('num_batch_splits', 1)
train_workers = cfg['dataset'].get('train_workers', None)
test_workers = cfg['dataset'].get('test_workers', None)
if total_batch_size % batch_splits != 0:
    raise ValueError('num_batch_splits must divide batch_size evenly')
split_batch_size = total_batch_size // batch_splits
# Load data with split batch size, which will be combined to the total batch size in tfprocess.
ChunkParser.BATCH_SIZE = split_batch_size

value_focus_min = cfg['training'].get('value_focus_min', 1)
value_focus_slope = cfg['training'].get('value_focus_slope', 0)

root_dir = os.path.join(cfg['training']['path'], cfg['name'])
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
tfprocess = TFProcess(cfg)
experimental_reads = max(2, mp.cpu_count() - 2) // 2
extractor = select_extractor(tfprocess.INPUT_MODE)

if experimental_parser and (value_focus_min != 1
                            or value_focus_slope != 0):
    raise ValueError(
        'Experimental parser does not support non-default value \
                        focus parameters.')

def read(x):
    return tf.data.FixedLengthRecordDataset(
        x,
        8308,
        compression_type='GZIP',
        num_parallel_reads=experimental_reads)

if experimental_parser:
    train_dataset = tf.data.Dataset.from_tensor_slices(train_chunks).shuffle(len(train_chunks)).repeat().batch(256)\
                        .interleave(read, num_parallel_calls=2)\
                        .batch(SKIP_MULTIPLE*SKIP).map(semi_sample).unbatch()\
                        .shuffle(shuffle_size)\
                        .batch(split_batch_size).map(extractor)
else:
    train_parser = ChunkParser(train_chunks,
                                tfprocess.INPUT_MODE,
                                shuffle_size=shuffle_size,
                                sample=SKIP,
                                batch_size=ChunkParser.BATCH_SIZE,
                                value_focus_min=value_focus_min,
                                value_focus_slope=value_focus_slope,
                                workers=train_workers)
    train_dataset = tf.data.Dataset.from_generator(
        train_parser.parse,
        output_types=(tf.string, tf.string, tf.string, tf.string,
                        tf.string))
    train_dataset = train_dataset.map(ChunkParser.parse_function)

shuffle_size = int(shuffle_size * (1.0 - train_ratio))
if experimental_parser:
    test_dataset = tf.data.Dataset.from_tensor_slices(test_chunks).shuffle(len(test_chunks)).repeat().batch(256)\
                        .interleave(read, num_parallel_calls=2)\
                        .batch(SKIP_MULTIPLE*SKIP).map(semi_sample).unbatch()\
                        .shuffle(shuffle_size)\
                        .batch(split_batch_size).map(extractor)
else:
    # no value focus for test_parser
    test_parser = ChunkParser(test_chunks,
                                tfprocess.INPUT_MODE,
                                shuffle_size=shuffle_size,
                                sample=SKIP,
                                batch_size=ChunkParser.BATCH_SIZE,
                                workers=test_workers)
    test_dataset = tf.data.Dataset.from_generator(
        test_parser.parse,
        output_types=(tf.string, tf.string, tf.string, tf.string,
                        tf.string))
    test_dataset = test_dataset.map(ChunkParser.parse_function)
validation_dataset = None