#!/usr/bin/env python3


import pickle
import numpy as np
import tensorflow as tf
import argparse

import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import lib
import lib.task.seq2seq.models.transformer_lrp as tr

BATCH_SIZE=50

def translate(args):
    #Load vocabularies
    inp_voc = pickle.load(open(args.ivoc, 'rb'))
    out_voc = pickle.load(open(args.ovoc, 'rb'))

    #Create model
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99, allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    model = tr.Model('mod', inp_voc, out_voc, inference_mode='fast', **args.hp)

    #Load checkpoint
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    lib.train.saveload.load(args.checkpoint, var_list)

    #Load source file
    test_src = [ l.strip() for l in open(args.input).readlines() ]

    #Manual, inefficient, in-memory batcher
    num_batches=len(test_src)//BATCH_SIZE
    if len(test_src) % BATCH_SIZE > 0:
        num_batches+=1

    for batch_id in range(num_batches):
        batch_data=test_src[batch_id*BATCH_SIZE:(batch_id+1)*BATCH_SIZE]
        translations=model.translate_lines(batch_data)
        for t in translations:
            print(t)

def TRANSLATE_add_params(p):
    #eval_arg = lambda x: eval(x, locals(), globals())
    p.add_argument('--hp', default="{}")
    p.add_argument('--ivoc', required=True)
    p.add_argument('--ovoc', required=True)
    p.add_argument('--checkpoint',help="Path to checkpoint", required=True)
    p.add_argument('--input',help="Path to input file", required=True)
    p.add_argument('--end-of-params', action='store_true', default=False)


def main():
    # Create parser.
    p = argparse.ArgumentParser('translate.py')

    # Add subcommands.
    train = TRANSLATE_add_params(p)

    # Parse.
    args = p.parse_args()
    args.hp=eval(args.hp, locals(), globals())

    translate(args)


if __name__ == '__main__':
    main()
