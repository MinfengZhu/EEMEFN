import time
import glob,os
import numpy as np
import h5py
import fnmatch
import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from numpy import linalg as LA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import rawpy
import scipy.io
import imageio

from dataset import SID_dataset as SID_dataset
from network import Netowrk as Netowrk


###############
## Test
###############

def test(set_name='Sony', stage='Merge', checkpoint=''):
    output_dir = "../result/" + set_name + "/"
    psnr_summary = []
    ssim_summary = []
    sess = tf.Session()

    data = SID_dataset(set_name=set_name, stage=stage, debug=True, patch_size=256)
    data.save_training_data_into_memory()
    test_dataset = data.get_test_dataset()
    iterator = test_dataset.make_one_shot_iterator()
    batch_image, batch_gt_image = iterator.get_next()

    in_image1 = tf.placeholder(tf.float32, [None, None, None, data.C_packed])
    in_image2 = tf.placeholder(tf.float32, [None, None, None, data.C_packed])
    gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
    network = Netowrk(set_name=set_name)
    input = tf.concat([in_image1, in_image2], 3)
    #gen_image = network.SID(input)
    gen_image, edge, hdr_image, edges = network.main(in_image1, in_image2)
    if set_name == 'Sony':
        edge = tf.depth_to_space(edge, 2)
    elif set_name == 'Fuji':
        edge = tf.depth_to_space(edge, 3)

    tf.train.init_from_checkpoint(checkpoint, {'hdr_fusion/': 'hdr_fusion/'})
    tf.train.init_from_checkpoint(checkpoint, {'edge/': 'edge/'})
    tf.train.init_from_checkpoint(checkpoint, {'merge/': 'merge/'})
    ssim1 = tf.image.ssim(gen_image, gt_image, max_val=1.0)
    psnr1 = tf.image.psnr(gen_image, gt_image, max_val=1.0)
    ssim2 = tf.image.ssim(hdr_image, gt_image, max_val=1.0)
    psnr2 = tf.image.psnr(hdr_image, gt_image, max_val=1.0)

    sess.run(tf.global_variables_initializer())
    pbar = tqdm(total=len(data.test_data_filenames))

    for i in range(len(data.test_data_filenames)):
        input_img, gt_image_ = sess.run([batch_image, batch_gt_image])
        input_img1 = input_img['in_raw1']
        input_img2 = input_img['in_raw2']
        input_img3 = input_img['in_raw3']
        id = input_img['id'][0].decode('UTF-8')

        fn = input_img['file_name'][0].decode('UTF-8')
        ratio = int(input_img['ratio'][0])
        in_exposure = input_img['in_exposure'][0]

        output_full, output_hdr_full, gt_full, edge_full, ssim1_, psnr1_, ssim2_, psnr2_ = sess.run(
            [gen_image, hdr_image, gt_image, edge, ssim1, psnr1, ssim2, psnr2],
            feed_dict={in_image1: input_img2, in_image2: input_img3, gt_image: gt_image_})

        in_files = glob.glob(data.short_dir + fn + data.image_format)
        raw = rawpy.imread(in_files[0])
        raw = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        raw = np.float32(raw / 65535.0)
        scale_full = raw * np.mean(gt_full) / np.mean(raw)
        input1_full = raw * ratio
        input2_full = raw * ratio / 2
        input3_full = raw

        output_full = np.minimum(np.maximum(output_full, 0), 1)
        output_full = output_full[0, :, :, :]

        output_hdr_full = np.minimum(np.maximum(output_hdr_full, 0), 1)
        output_hdr_full = output_hdr_full[0, :, :, :]

        gt_full = gt_full[0, :, :, :]

        edge_full = edge_full[0, :, :, :]
        edge_full = np.concatenate((edge_full, edge_full, edge_full), axis=2)

        # scipy.misc.toimage(output_full * 255, high=255, low=0, cmin=0, cmax=255).save(
        #     output_dir + '{}_{}_{:.3f}_{}_{}_out.png'.format(id, ratio, in_exposure, psnr1_, ssim1_))
        # scipy.misc.toimage((1-edge_full) * 255, high=255, low=0, cmin=0, cmax=255).save(
        #     output_dir + '{}_{}_out_edge.png'.format(id, ratio))
        # scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
        #     output_dir + '{}_{}_gt.png'.format(id, ratio))



        psnr_summary.append(psnr1_[0])
        ssim_summary.append(ssim1_[0])
        pbar.update(1)
        #break
    pbar.close()
    print("PSNR: " + str(np.mean(psnr_summary)))
    print("SSIM: " + str(np.mean(ssim_summary)))
    sess.close()
    tf.reset_default_graph()


test(set_name='Sony', checkpoint='../result/Sony_merge/model.ckpt-800000')
test(set_name='Fuji', checkpoint='../result/Fuji_merge/model.ckpt-800000')
