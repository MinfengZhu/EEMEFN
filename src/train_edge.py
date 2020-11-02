from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import numpy as np
from PIL import Image
tf.logging.set_verbosity(tf.logging.INFO)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from dataset import SID_dataset as SID_dataset
from network import Netowrk as Netowrk

###############
## Data
##############
set_name = 'Sony'
data = SID_dataset(set_name=set_name, stage='EDGE', debug=False, patch_size=512)
data.save_training_data_into_memory()

def train_input_fn():
    """An input function for training"""
    dataset = data.get_edge_training_dataset()
    return dataset

def test_input_fn():
    dataset = data.get_edge_test_dataset()
    return dataset

###############
## Model
##############
model_dir = "../result/" + set_name + "_20190628_edge"

def model_fn(features, labels, mode):
    input = features['image']
    input = tf.space_to_depth(input, data.ratio_packed)
    gt_mask = features['mask']
    gt_mask = tf.space_to_depth(gt_mask, data.ratio_packed)
    gt_image = labels
    gt_edge_ = tf.space_to_depth(gt_image, data.ratio_packed)

    network = Netowrk(set_name=set_name)
    [s1_out, s2_out, s3_out, s4_out, s5_out, fuse_out] = network.edge(input)
    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for v in vars: print(v)

    loss = 0 #network.edge_loss(fuse_out, gt_edge_, gt_mask)
    for gen_edge in [s1_out, s2_out, s3_out, s4_out, s5_out, fuse_out]:
        loss += network.edge_loss(gen_edge, gt_edge_, gt_mask)



    gen_edge = tf.sigmoid(fuse_out)
    gen_edge = tf.depth_to_space(gen_edge, data.ratio_packed)
    gt_edge = gt_image

    out_image_cut = tf.cast(tf.minimum(tf.maximum(gen_edge * 255, 0), 255), tf.uint8)
    gt_image_cut = tf.cast(tf.minimum(tf.maximum(gt_edge * 255, 0), 255), tf.uint8)

    tf.train.init_from_checkpoint('../result/Sony_20190311_edge/model.ckpt-785138', {'edge/': 'edge/'})

    if mode == tf.estimator.ModeKeys.EVAL:
        summary_hook = tf.train.SummarySaverHook(
            500,
            output_dir=model_dir + '/eval',
            summary_op=[
                tf.summary.image('result_gt_image', tf.concat((out_image_cut, gt_image_cut), axis=2), max_outputs=1)]
        )
        eval_metric_ops = {'SSIM': tf.metrics.mean(tf.image.ssim(gen_edge, gt_image, max_val=1.0)),
                           'PSN': tf.metrics.mean(tf.image.psnr(gen_edge, gt_image, max_val=1.0)),
                           }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops, evaluation_hooks=[summary_hook])

    # if mode == tf.estimator.ModeKeys.PREDICT:
    #
    #     predictions = {"gen_image": gen_image
    #                    }
    #     return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Optimizer
        learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(), [160 * 2500], [1e-4, 1e-5])
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=tf.train.get_global_step())

        # Summary
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.image('result_gt_image', tf.concat((out_image_cut, gt_image_cut), axis=2), max_outputs=1)
        tf.summary.scalar('PSN', tf.reduce_mean(tf.image.psnr(gen_edge, gt_image, max_val=1.0)))
        tf.summary.scalar('SSIM', tf.reduce_mean(tf.image.ssim(gen_edge, gt_image, max_val=1.0)))
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


###############
## Estimator
##############
distribution = tf.contrib.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(
    train_distribute=distribution,
    model_dir=model_dir,
    save_checkpoints_secs=30 * 60,  # Save checkpoints every 30 minutes.
    keep_checkpoint_max=1,  # Retain the 10 most recent checkpoints.
    save_summary_steps=100000
)
estimator = tf.estimator.Estimator(model_fn=model_fn, config=config)

###############
## Train
###############
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=160 * 5000)#
eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn, throttle_secs=1000, steps=data.n_test)
st = time.time()
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
print("Time=%.3f" % (time.time() - st))

st = time.time()
eval = estimator.evaluate(input_fn=test_input_fn, steps=data.n_test)
print("Time=%.3f" % (time.time() - st))
print(eval)



