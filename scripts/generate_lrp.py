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

from lib.ops.record_activations import recording_activations
from lib.layers.basic import dropout_scope
from lib.ops import record_activations as rec
from lib.layers.lrp import LRP

BATCH_SIZE=50


def get_topk_logits_selector(logits, k=3):
    """ takes logits[batch, nout, voc_size] and returns a mask with ones at k largest logits """
    topk_logit_indices = tf.nn.top_k(logits, k=k).indices
    indices = tf.stack([
        tf.range(tf.shape(logits)[0] * tf.shape(logits)[1] * k) // (tf.shape(logits)[1] * k),
        (tf.range(tf.shape(logits)[0] * tf.shape(logits)[1] * k) // k) % tf.shape(logits)[1],
        tf.reshape(topk_logit_indices, [-1])
    ], axis=1)
    ones = tf.ones(shape=(tf.shape(indices)[0],))
    return tf.scatter_nd(indices, ones, shape=tf.shape(logits))

def generate_lrp(args):
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
    test_src = [ l.strip() for l in open(args.input_src).readlines() ]
    test_dst = [ l.strip() for l in open(args.input_dst).readlines() ]

    #Create feed_dict
    #feed_dict = model.make_feed_dict(zip(test_src[:3], test_dst[:3]))
    #See original code above: maybe we should batch?
    feed_dict = model.make_feed_dict(zip(test_src, test_dst))
    ph = lib.task.seq2seq.data.make_batch_placeholder(feed_dict)
    feed = {ph[key]: feed_dict[key] for key in feed_dict}

    target_position = tf.placeholder(tf.int32, [])
    with rec.recording_activations() as saved_activations, dropout_scope(False):
        rdo = model.encode_decode(ph, is_train=False)
        logits = model.loss._rdo_to_logits(rdo)
        out_mask = tf.one_hot(target_position, depth=tf.shape(logits)[1])[None, :, None]

        top1_logit = get_topk_logits_selector(logits, k=1) * tf.nn.softmax(logits)
        top1_prob = tf.reduce_sum(top1_logit, axis=-1)[0]

        R_ = get_topk_logits_selector(logits, k=1) * out_mask
        R = model.loss._rdo_to_logits.relprop(R_)
        R = model.transformer.relprop_decode(R)

        R_out = tf.reduce_sum(abs(R['emb_out']), axis=-1)
        R_inp = tf.reduce_sum(abs(model.transformer.relprop_encode(R['enc_out'])), axis=-1)


    result = []

    for elem in zip(test_src, test_dst):
        #print(len(result))
        src = elem[0].strip()
        dst = elem[1].strip()
        dst_words = len(dst.split()) + 1
        feed_dict = model.make_feed_dict(zip([src], [dst]))
        feed = {ph[key]: feed_dict[key] for key in feed_dict}

        inp_lrp = []
        out_lrp = []
        for token_pos in range(feed_dict['out'].shape[1]):
            feed[target_position] = token_pos
            res_inp, res_out = sess.run((R_inp, R_out), feed)
            inp_lrp.append(res_inp[0])
            out_lrp.append(res_out[0])
        result.append({'src': src, 'dst': dst,
                       'inp_lrp': np.array(inp_lrp), 'out_lrp': np.array(out_lrp)
                      })
    pickle.dump(result, open(args.output, 'wb'))


    #Manual, inefficient, in-memory batcher
    #num_batches=len(test_src)//BATCH_SIZE
    #if len(test_src) % BATCH_SIZE > 0:
    #    num_batches+=1

    #for batch_id in range(num_batches):
    #    batch_data=test_src[batch_id*BATCH_SIZE:(batch_id+1)*BATCH_SIZE]
    #    translations=model.translate_lines(batch_data)
    #    for t in translations:
    #        print(t)

def TRANSLATE_add_params(p):
    #eval_arg = lambda x: eval(x, locals(), globals())
    p.add_argument('--hp', default="{}")
    p.add_argument('--ivoc', required=True)
    p.add_argument('--ovoc', required=True)
    p.add_argument('--checkpoint',help="Path to checkpoint", required=True)
    p.add_argument('--input-src',help="Path to input file (source language)", required=True)
    p.add_argument('--input-dst',help="Path to input file (target language)", required=True)
    p.add_argument('--output',help="Path to output file", required=True)
    p.add_argument('--end-of-params', action='store_true', default=False)


def main():
    # Create parser.
    p = argparse.ArgumentParser('generate_lrp.py')

    # Add subcommands.
    train = TRANSLATE_add_params(p)

    # Parse.
    args = p.parse_args()
    args.hp=eval(args.hp, locals(), globals())

    generate_lrp(args)


if __name__ == '__main__':
    main()
