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
data = SID_dataset(set_name=set_name, stage='MERGE', debug=False, patch_size=512)
data.save_training_data_into_memory()

def train_input_fn():
    """An input function for training"""
    dataset = data.get_training_dataset()
    return dataset

def test_input_fn():
    dataset = data.get_test_dataset()
    return dataset

###############
## Model
##############
model_dir = "../result/" + set_name + "merge"

def model_fn(features, labels, mode):
    input1 = features['in_raw1']
    input2 = features['in_raw2']
    input3 = features['in_raw3']
    gt_edge = features['edge']
    gt_mask = features['mask']
    gt_image = labels

    network = Netowrk(set_name=set_name)
    gen_image, edge, hdr_image, edges = network.main(input2, input3)
    if set_name == 'Sony':
        edge = tf.depth_to_space(edge, 2)
    elif set_name == 'Fuji':
        edge = tf.depth_to_space(edge, 3)

    tf.train.init_from_checkpoint('../result/Sony_fusion/model.ckpt-756788', {'hdr_fusion/': 'hdr_fusion/'})
    tf.train.init_from_checkpoint('../result/Sony_edge/model.ckpt-785138', {'edge/': 'edge/'})

    #tf.train.init_from_checkpoint('../result/Fuji_fusion/model.ckpt-762186', {'hdr_fusion/': 'hdr_fusion/'})
    #tf.train.init_from_checkpoint('../result/Fuji_edge/model.ckpt-800000', {'edge/': 'edge/'})


    loss_edge = 0
    for gen_edge in edges:
        loss_edge += network.edge_loss(gen_edge, tf.space_to_depth(gt_edge, data.ratio_packed), tf.space_to_depth(gt_mask, data.ratio_packed))
    loss = tf.reduce_mean(tf.abs(gt_image - gen_image))

    #vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    #for v in vars: print(v)

    if mode == tf.estimator.ModeKeys.EVAL:
        #gen_image = tf.Print(gen_image, [tf.shape(gen_image)])
        out_image_cut = tf.cast(tf.minimum(tf.maximum(gen_image * 255, 0), 255), tf.uint8)
        gt_image_cut = tf.cast(tf.minimum(tf.maximum(gt_image * 255, 0), 255), tf.uint8)
        out_edge_cut = tf.cast(tf.minimum(tf.maximum(edge * 255, 0), 255), tf.uint8)
        out_edge_cut = tf.concat((out_edge_cut, out_edge_cut, out_edge_cut), axis=3)
        summary_hook = tf.train.SummarySaverHook(
            500,
            output_dir=model_dir + '/eval',
            summary_op=[
                tf.summary.image('result_gt_image', tf.concat((out_image_cut, out_edge_cut, gt_image_cut), axis=2), max_outputs=1)]
        )
        eval_metric_ops = {'SSIM': tf.metrics.mean(tf.image.ssim(gen_image, gt_image, max_val=1.0)),
                           'PSN': tf.metrics.mean(tf.image.psnr(gen_image, gt_image, max_val=1.0)),
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
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='merge')
        vars_edge = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='edge')
        # for v in vars: print(v)
        train_op_merge = tf.train.AdamOptimizer(learning_rate=learning_rate) \
            .minimize(loss, global_step=tf.train.get_global_step(), var_list=vars)
        train_op_edge = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_edge, var_list=vars_edge)
        #train_op = tf.group(train_op_merge, train_op_edge)
        train_op = tf.group(train_op_merge)
        #train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss+loss_edge, global_step=tf.train.get_global_step())

        # Summary
        tf.summary.scalar('learning_rate', learning_rate)
        out_image_cut = tf.cast(tf.minimum(tf.maximum(gen_image * 255, 0), 255), tf.uint8)
        gt_image_cut = tf.cast(tf.minimum(tf.maximum(gt_image * 255, 0), 255), tf.uint8)
        tf.summary.scalar('PSN', tf.reduce_mean(tf.image.psnr(gen_image, gt_image, max_val=1.0)))
        tf.summary.scalar('SSIM', tf.reduce_mean(tf.image.ssim(gen_image, gt_image, max_val=1.0)))
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


###############
## Estimator
##############
config = tf.estimator.RunConfig(
    model_dir=model_dir,
    save_checkpoints_secs=30 * 60,  # Save checkpoints every 30 minutes.
    keep_checkpoint_max=1,  # Retain the 10 most recent checkpoints.
    save_summary_steps=10000,
    log_step_count_steps=500, # log training information per 500 steps
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
eval = estimator.evaluate(input_fn=test_input_fn)
print("Time=%.3f" % (time.time() - st))
print(eval)



