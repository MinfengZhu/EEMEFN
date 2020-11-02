from __future__ import division
import os
import tensorflow as tf
import numpy as np
import rawpy
import glob
import time
import random
from skimage import color
from skimage import feature
from tqdm import tqdm
import h5py
import fnmatch


class SID_dataset():
    def __init__(self, set_name, patch_size=512, stage='MF', debug=False):
        self.debug = debug
        self.set_name = set_name
        self.stage = stage
        if self.set_name == "Sony":
            self.image_format = ".ARW"
            self.dir = "../dataset/Sony/"
            self.short_dir = '../dataset/Sony/short/'
            self.long_dir = '../dataset/Sony/long/'
            self.short_file = '../dataset/Sony/sony_short.hdf5'
            self.long_file = '../dataset/Sony/sony_long.hdf5'
            self.short_data = []
            self.long_data = []
            self.n_input_train = 280
            self.n_gt_train = 161
            self.n_test = 88
            self.H = 2848
            self.W = 4256
            self.H_out = 2848
            self.W_out = 4256
            self.H_packed = 1424
            self.W_packed = 2128
            self.C_packed = 4
            self.ratio_packed = 2
            self.patch_size = patch_size
            self.black_level = 512
            if self.debug == False:
                self.n_items_in_memory = 280
            elif self.debug == True:
                self.n_items_in_memory = 0
            if self.stage == 'EDGE':
                self.n_items_in_memory = 0


            self.train_input_raw = np.zeros((self.n_items_in_memory, self.H_packed, self.W_packed , self.C_packed), dtype=np.uint16)
            self.train_input_raw_filename = []
            self.train_input_index = {}
            self.train_input_ratio = np.zeros((self.n_input_train), dtype=np.float32)
            self.train_gt_rgb = np.zeros((self.n_gt_train, self.H, self.W, 3), dtype=np.uint16)
            self.train_gt_edge = np.zeros((self.n_gt_train, self.H, self.W, 1), dtype=np.bool)
        elif self.set_name == "Fuji":
            self.image_format = ".RAF"
            self.dir = "../dataset/Fuji/"
            self.short_dir = '../dataset/Fuji/short/'
            self.long_dir = '../dataset/Fuji/long/'
            self.short_file = '../dataset/Fuji/fuji_short.hdf5'
            self.long_file = '../dataset/Fuji/fuji_long.hdf5'
            self.short_data = []
            self.long_data = []
            self.n_input_train = 286
            self.n_gt_train = 135
            self.n_test = 94
            self.H = 4032
            self.W = 6032
            self.H_out = 4032
            self.W_out = 6030
            self.H_packed = 1344
            self.W_packed = 2010
            self.C_packed = 9
            self.ratio_packed = 3
            self.patch_size = patch_size
            self.black_level = 1024
            if self.debug == False:
                self.n_items_in_memory = 286
            elif self.debug == True:
                self.n_items_in_memory = 0
            if self.stage == 'EDGE':
                self.n_items_in_memory = 0

            self.train_input_raw = np.zeros((self.n_items_in_memory, self.H_packed, self.W_packed, self.C_packed), dtype=np.uint16)
            self.train_input_raw_filename = []
            self.train_input_index = {}
            self.train_input_ratio = np.zeros((self.n_input_train), dtype=np.float32)
            self.train_gt_rgb = np.zeros((self.n_gt_train, self.H, self.W, 3), dtype=np.uint16)
            self.train_gt_edge = np.zeros((self.n_gt_train, self.H, self.W, 1), dtype=np.bool)


    def save_short_data_as_np_file(self):
        def read_Sony_Short_by_filepath(filepath):
            def preprocess_raw_image(image):
                image = np.expand_dims(image, axis=2)
                img_shape = image.shape
                H = img_shape[0]
                W = img_shape[1]
                image = np.concatenate([image[0:H:2, 0:W:2, :],
                                        image[0:H:2, 1:W:2, :],
                                        image[1:H:2, 1:W:2, :],
                                        image[1:H:2, 0:W:2, :],
                                        ], axis=2)
                return image

            raw = rawpy.imread(filepath)
            img = raw.raw_image_visible
            img = np.uint16(img)
            img = preprocess_raw_image(img)
            return img

        def read_Fuji_Short_by_filepath(filepath):
            def preprocess_raw_image(image):
                # get shape
                img_shape = image.shape
                H = (img_shape[0] // 6) * 6
                W = (img_shape[1] // 6) * 6

                out = np.zeros((H // 3, W // 3, 9))
                # 0 R
                out[0::2, 0::2, 0] = image[0:H:6, 0:W:6]
                out[0::2, 1::2, 0] = image[0:H:6, 4:W:6]
                out[1::2, 0::2, 0] = image[3:H:6, 1:W:6]
                out[1::2, 1::2, 0] = image[3:H:6, 3:W:6]

                # 1 G
                out[0::2, 0::2, 1] = image[0:H:6, 2:W:6]
                out[0::2, 1::2, 1] = image[0:H:6, 5:W:6]
                out[1::2, 0::2, 1] = image[3:H:6, 2:W:6]
                out[1::2, 1::2, 1] = image[3:H:6, 5:W:6]

                # 1 B
                out[0::2, 0::2, 2] = image[0:H:6, 1:W:6]
                out[0::2, 1::2, 2] = image[0:H:6, 3:W:6]
                out[1::2, 0::2, 2] = image[3:H:6, 0:W:6]
                out[1::2, 1::2, 2] = image[3:H:6, 4:W:6]

                # 4 R
                out[0::2, 0::2, 3] = image[1:H:6, 2:W:6]
                out[0::2, 1::2, 3] = image[2:H:6, 5:W:6]
                out[1::2, 0::2, 3] = image[5:H:6, 2:W:6]
                out[1::2, 1::2, 3] = image[4:H:6, 5:W:6]

                # 5 B
                out[0::2, 0::2, 4] = image[2:H:6, 2:W:6]
                out[0::2, 1::2, 4] = image[1:H:6, 5:W:6]
                out[1::2, 0::2, 4] = image[4:H:6, 2:W:6]
                out[1::2, 1::2, 4] = image[5:H:6, 5:W:6]

                out[:, :, 5] = image[1:H:3, 0:W:3]
                out[:, :, 6] = image[1:H:3, 1:W:3]
                out[:, :, 7] = image[2:H:3, 0:W:3]
                out[:, :, 8] = image[2:H:3, 1:W:3]

                return out

            raw = rawpy.imread(filepath)
            img = raw.raw_image_visible
            img = preprocess_raw_image(img)
            img = np.uint16(img)
            return img

        # # creat h5file
        # h5file = h5py.File(self.short_file, "w")
        # # get IDs
        # fns = glob.glob(self.long_dir + '*' + self.image_format)
        # ids = sorted([int(os.path.basename(fn)[0:5]) for fn in fns])
        # # progress bar
        # pbar = tqdm(total=len(ids))
        # #for each ID
        # for ind in range(len(ids)):
        #     # get the path from image id
        #     train_id = ids[ind]
        #     # short exposure image
        #     short_files = glob.glob(self.short_dir + '%05d_*' % train_id + self.image_format)
        #     for short_path in short_files:
        #         short_fn = os.path.basename(short_path)
        #         if self.set_name == "Sony":
        #             raw = read_Sony_Short_by_filepath(short_path)
        #         h5file.create_dataset(short_fn[:-4], data=raw, chunks=True)
        #         pbar.set_description("Processing Short Exposure Data {}: ".format(short_fn))
        #     pbar.update(1)
        # pbar.close()

        # creat h5file
        h5file = h5py.File(self.short_file, "w")
        # get IDs
        fns = glob.glob(self.short_dir + '*' + self.image_format)
        # progress bar
        pbar = tqdm(total=len(fns), ascii=' >>>>>-')
        for idx, path in enumerate(fns):
            if self.set_name == "Sony":
                raw = read_Sony_Short_by_filepath(path)
            elif self.set_name == "Fuji":
                raw = read_Fuji_Short_by_filepath(path)
            h5file.create_dataset(os.path.basename(path)[:-4], data=raw, chunks=True)
            pbar.set_description("Processing Short Exposure Data {}: ".format(os.path.basename(path)))
            pbar.update(1)
        pbar.close()

    def save_long_data_as_np_file(self):
        def read_Long_by_filepath(filepath):
            raw = rawpy.imread(filepath)
            img_rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            img_rgb = np.uint16(img_rgb)
            return img_rgb

        # creat h5file
        h5file = h5py.File(self.long_file, "w")
        # get long exposure image
        long_fns = glob.glob(self.long_dir + '*' + self.image_format)
        # progress bar
        pbar = tqdm(total=len(long_fns), ascii=' >>>>>-')
        for idx, path in enumerate(long_fns):
            img_rgb = read_Long_by_filepath(path)
            h5file.create_dataset(os.path.basename(path)[:-4], data=img_rgb)
            img_rgb = np.float32(img_rgb)
            img_rgb = img_rgb / 65535.0
            img_edge = color.rgb2gray(img_rgb)
            img_edge = feature.canny(img_edge, sigma=0.5)
            img_edge = np.expand_dims(img_edge, axis=2)
            h5file.create_dataset(os.path.basename(path)[:-4]+".edge", data=img_edge)
            pbar.set_description("Processing Long Exposure Data {}: ".format(os.path.basename(path)))
            pbar.update(1)
        pbar.close()

    def save_training_data_into_memory(self):
        self.short_data = h5py.File(self.short_file, 'r')
        self.long_data = h5py.File(self.long_file, 'r')
        short_keys = list(self.short_data.keys())
        long_keys = list(self.long_data.keys())
        # progress bar
        pbar = tqdm(total=self.n_input_train, ascii=' >>>>>-')
        # get train IDs
        train_fns = [name for name in long_keys if fnmatch.fnmatch(name, '0*s')]
        train_ids = sorted([int(train_fn[0:5]) for train_fn in train_fns])
        if self.debug == True:
            train_ids = train_ids[:10]
        count = 0
        for ind in range(len(train_ids)):
            train_id = train_ids[ind]
            #ground truth image
            gt_files = [name for name in long_keys if fnmatch.fnmatch(name, '%05d_00*s' % train_id)]
            gt_fn = gt_files[0]
            self.train_gt_rgb[ind, :, :, :] = self.long_data[gt_fn][:]
            self.train_gt_edge[ind, :, :, :] = self.long_data[gt_fn+".edge"][:]
            # training image
            self.train_input_index[str(ind)] = {}
            for exposure in [0.1, 0.04, 0.033]:
                fn_names = short_keys
                in_files = [name for name in fn_names if fnmatch.fnmatch(name, '{:05d}_*{}s'.format(train_id, exposure))]
                if len(in_files) > 0:
                    in_fn = os.path.basename(in_files[0])
                    in_exposure = float(in_fn[9:-1])
                    gt_exposure = float(gt_fn[9:-1])
                    ratio = min(gt_exposure / in_exposure, 300)
                    self.train_input_index[str(ind)][str(ratio)] = []
                    self.train_input_index[str(ind)][str(ratio)].append(count)
                    self.train_input_raw_filename.append(in_files[0])
                    self.train_input_ratio[count] = ratio
                    if count < self.n_items_in_memory:
                        self.train_input_raw[count, :, :, :] = self.short_data[in_files[0]][:]
                    count += 1
                    pbar.set_description("Reading Data into Memory: " + in_files[0])
                    pbar.update(1)
        pbar.close()
        print(self.train_input_index)

    def get_training_dataset(self):
        def gen():
            for index in np.random.permutation(self.n_gt_train):
                # index
                if self.debug == True:
                    index = 0
                exposure_index = random.choice(list(self.train_input_index[str(index)].keys()))
                img_idx = random.choice(self.train_input_index[str(index)][exposure_index])
                if self.debug == True:
                    img_idx = 0

                # Crop
                ps = self.patch_size
                H = self.H_packed
                W = self.W_packed
                xx = np.random.randint(0, W - ps)
                yy = np.random.randint(0, H - ps)
                if self.debug == True:
                    xx = 500
                    yy = 500

                # read image
                if img_idx < self.n_items_in_memory:
                    input_raw = self.train_input_raw[img_idx, yy:yy + ps, xx:xx + ps, :]
                else:
                    input_raw = self.short_data[self.train_input_raw_filename[img_idx]][yy:yy + ps, xx:xx + ps, :]
                r = self.ratio_packed
                ratio = self.train_input_ratio[img_idx]
                gt_rgb = self.train_gt_rgb[index, yy * r:yy * r + ps * r, xx * r:xx * r + ps * r, :]
                gt_edge = self.train_gt_edge[index, yy * r:yy * r + ps * r, xx * r:xx * r + ps * r, :]
                num_positive = np.count_nonzero(gt_edge == 1)
                num_negative = np.count_nonzero(gt_edge == 0)
                gt_mask = np.float32(gt_edge)
                gt_mask[gt_mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
                gt_mask[gt_mask == 0] = 1.1 * num_positive / (num_positive + num_negative)

                if np.random.randint(2, size=1)[0] == 1:  # random flip
                    input_raw = np.flip(input_raw, axis=0)
                    gt_rgb = np.flip(gt_rgb, axis=0)
                    gt_edge = np.flip(gt_edge, axis=0)
                    gt_mask = np.flip(gt_mask, axis=0)

                if np.random.randint(2, size=1)[0] == 1:
                    input_raw = np.flip(input_raw, axis=1)
                    gt_rgb = np.flip(gt_rgb, axis=1)
                    gt_edge = np.flip(gt_edge, axis=1)
                    gt_mask = np.flip(gt_mask, axis=1)

                if np.random.randint(2, size=1)[0] == 1:  # random transpose
                    input_raw = np.transpose(input_raw, (1, 0, 2))
                    gt_rgb = np.transpose(gt_rgb, (1, 0, 2))
                    gt_edge = np.transpose(gt_edge, (1, 0, 2))
                    gt_mask = np.transpose(gt_mask, (1, 0, 2))

                yield (ratio, input_raw, gt_rgb, gt_edge, gt_mask) # image_rgb_patch

        def map_func(ratio, image_patch, gt_image_patch, gt_edge_patch, gt_mask_patch):  #
            # normalization
            gt_image_patch = tf.cast(gt_image_patch, tf.float32)
            gt_image_patch = gt_image_patch / 65535.0
            gt_edge_patch = tf.cast(gt_edge_patch, tf.float32)
            image_patch = tf.cast(image_patch, tf.float32)
            image_patch = tf.maximum(image_patch - self.black_level, 0) / (16383 - self.black_level)
            image_patch1 = image_patch * ratio
            image_patch1 = tf.minimum(image_patch1, 1)
            image_patch2 = image_patch * ratio / 2
            image_patch2 = tf.minimum(image_patch2, 1)
            image_patch3 = image_patch

            # set shape
            image_patch1.set_shape([self.patch_size, self.patch_size, self.C_packed])
            image_patch2.set_shape([self.patch_size, self.patch_size, self.C_packed])
            image_patch3.set_shape([self.patch_size, self.patch_size, self.C_packed])
            gt_image_patch.set_shape([self.patch_size * self.ratio_packed, self.patch_size * self.ratio_packed, 3])
            gt_edge_patch.set_shape([self.patch_size * self.ratio_packed, self.patch_size * self.ratio_packed, 1])
            gt_mask_patch.set_shape([self.patch_size * self.ratio_packed, self.patch_size * self.ratio_packed, 1])

            features = {'in_raw1': image_patch1, 'in_raw2': image_patch2, 'in_raw3': image_patch3, 'edge': gt_edge_patch, 'mask': gt_mask_patch}
            return features, gt_image_patch

        dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.uint16, tf.uint16, tf.bool, tf.float32))#,
        dataset = dataset.shuffle(buffer_size=10).repeat()
        dataset = dataset.map(map_func=map_func, num_parallel_calls=8)
        dataset = dataset.batch(batch_size=1, drop_remainder=True)
        dataset = dataset.prefetch(10)
        return dataset

    def get_test_dataset(self):
        short_keys = list(self.short_data.keys())
        long_keys = list(self.long_data.keys())
        test_fns = [name for name in long_keys if fnmatch.fnmatch(name, '1*s')]
        test_ids = [int(test_fn[0:5]) for test_fn in test_fns]
        #print(test_ids)
        self.test_data_filenames = []
        for test_id in test_ids:
            for exposure in [0.1, 0.04, 0.033]:
                in_files = [name for name in short_keys if fnmatch.fnmatch(name, '{:05d}_*{}s'.format(test_id, exposure))]
                if len(in_files) > 0:
                    in_path = in_files[0]
                    gt_files = [name for name in long_keys if fnmatch.fnmatch(name, '%05d_00*s' % test_id)]
                    gt_path = gt_files[0]
                    self.test_data_filenames.append([in_path, gt_path])
        #print(self.test_data_filenames)
        print(len(self.test_data_filenames))

        def gen():
            for i in range(len(self.test_data_filenames)):
                in_path, gt_path = self.test_data_filenames[i]
                in_fn = os.path.basename(in_path)
                gt_fn = os.path.basename(gt_path)
                in_exposure = float(in_fn[9:-1])
                gt_exposure = float(gt_fn[9:-1])
                ratio = min(gt_exposure / in_exposure, 300)
                in_raw = self.short_data[in_path][:]
                gt_rgb = self.long_data[gt_path][:]
                gt_edge = self.long_data[gt_path + ".edge"][:]
                num_positive = np.count_nonzero(gt_edge == 1)
                num_negative = np.count_nonzero(gt_edge == 0)
                gt_mask = np.float32(gt_edge)
                gt_mask[gt_mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
                gt_mask[gt_mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
                yield (gt_path[0:5], in_path, in_exposure, gt_exposure, ratio, in_raw, gt_rgb, gt_edge, gt_mask)

        def map(id, in_path, in_exposure, gt_exposure, ratio, in_raw, gt_rgb, gt_edge, gt_mask):
            # normalization
            gt_rgb = tf.cast(gt_rgb, tf.float32)
            gt_rgb = gt_rgb / 65535.0
            in_raw = tf.cast(in_raw, tf.float32)
            in_raw = tf.maximum(in_raw - self.black_level, 0) / (16383 - self.black_level)
            in_raw1 = in_raw * ratio
            in_raw1 = tf.minimum(in_raw1, 1)
            in_raw2 = in_raw * ratio / 2
            in_raw2 = tf.minimum(in_raw2, 1)
            in_raw3 = in_raw

            in_raw1.set_shape([self.H_packed, self.W_packed, self.C_packed])
            in_raw2.set_shape([self.H_packed, self.W_packed, self.C_packed])
            in_raw3.set_shape([self.H_packed, self.W_packed, self.C_packed])
            gt_rgb = gt_rgb[0:self.H_out, 0:self.W_out, :]
            gt_rgb.set_shape([self.H_out, self.W_out, 3])
            gt_edge = gt_edge[0:self.H_out, 0:self.W_out, :]
            gt_edge.set_shape([self.H_out, self.W_out, 1])
            gt_mask = gt_mask[0:self.H_out, 0:self.W_out, :]
            gt_mask.set_shape([self.H_out, self.W_out, 1])
            feature = {'in_raw1': in_raw1, 'in_raw2': in_raw2, 'in_raw3': in_raw3, 'edge': gt_edge, 'id': id, 'ratio': ratio, 'file_name': in_path, 'in_exposure': in_exposure, 'gt_exposure': gt_exposure, 'mask': gt_mask}  # 'in_rgb': in_rgb,
            return feature, gt_rgb

        dataset = tf.data.Dataset.from_generator(gen, (tf.string, tf.string, tf.float32, tf.float32, tf.float32, tf.uint16, tf.uint16, tf.bool, tf.float32))
        #dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1))
        dataset = dataset.map(map_func=map, num_parallel_calls=8)
        dataset = dataset.batch(batch_size=1, drop_remainder=True)
        dataset = dataset.prefetch(10)
        return dataset

    def get_edge_training_dataset(self):
        def gen():
            for index in np.random.permutation(self.n_gt_train):
                # index
                if self.debug == True:
                    index = 0
                # Crop
                ps = self.patch_size
                H = self.H_packed
                W = self.W_packed
                xx = np.random.randint(0, W - ps)
                yy = np.random.randint(0, H - ps)
                if self.debug == True:
                    xx = 500
                    yy = 500
                r = self.ratio_packed
                gt_rgb = self.train_gt_rgb[index, yy * r:yy * r + ps * r, xx * r:xx * r + ps * r, :]
                gt_edge = self.train_gt_edge[index, yy * r:yy * r + ps * r, xx * r:xx * r + ps * r, :]
                num_positive = np.count_nonzero(gt_edge == 1)
                num_negative = np.count_nonzero(gt_edge == 0)
                gt_mask = np.float32(gt_edge)
                gt_mask[gt_mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
                gt_mask[gt_mask == 0] = 1.1 * num_positive / (num_positive + num_negative)

                if np.random.randint(2, size=1)[0] == 1:  # random flip
                    gt_rgb = np.flip(gt_rgb, axis=0)
                    gt_edge = np.flip(gt_edge, axis=0)
                    gt_mask = np.flip(gt_mask, axis=0)

                if np.random.randint(2, size=1)[0] == 1:
                    gt_rgb = np.flip(gt_rgb, axis=1)
                    gt_edge = np.flip(gt_edge, axis=1)
                    gt_mask = np.flip(gt_mask, axis=1)

                if np.random.randint(2, size=1)[0] == 1:  # random transpose
                    gt_rgb = np.transpose(gt_rgb, (1, 0, 2))
                    gt_edge = np.transpose(gt_edge, (1, 0, 2))
                    gt_mask = np.transpose(gt_mask, (1, 0, 2))

                yield (gt_rgb, gt_mask, gt_edge)  # image_rgb_patch

        def map_func(image_patch, gt_mask_patch, gt_edge_patch):  #
            # normalization
            image_patch = tf.cast(image_patch, tf.float32)
            image_patch = image_patch / 65535.0
            gt_edge_patch = tf.cast(gt_edge_patch, tf.float32)

            # set shape
            r = self.ratio_packed
            image_patch.set_shape([self.patch_size * r, self.patch_size * r, 3])
            gt_mask_patch.set_shape([self.patch_size * r, self.patch_size * r, 1])
            gt_edge_patch.set_shape([self.patch_size * r, self.patch_size * r, 1])

            features = {'image': image_patch, 'mask': gt_mask_patch}
            return features, gt_edge_patch

        dataset = tf.data.Dataset.from_generator(gen, (tf.uint16, tf.float32, tf.bool))  # ,
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=2))
        dataset = dataset.map(map_func=map_func, num_parallel_calls=8)
        dataset = dataset.batch(batch_size=1, drop_remainder=True)
        dataset = dataset.prefetch(10)
        return dataset

    def get_edge_test_dataset(self):
        short_keys = list(self.short_data.keys())
        long_keys = list(self.long_data.keys())
        test_fns = [name for name in long_keys if fnmatch.fnmatch(name, '1*s')]
        test_ids = [int(test_fn[0:5]) for test_fn in test_fns]
        self.test_data_filenames = []
        for test_id in test_ids:
            for exposure in [0.1, 0.04, 0.033]:
                in_files = [name for name in short_keys if fnmatch.fnmatch(name, '{:05d}_*{}s'.format(test_id, exposure))]
                if len(in_files) > 1:
                    in_path1 = in_files[0]
                    in_path2 = in_files[1]
                    gt_files = [name for name in long_keys if fnmatch.fnmatch(name, '%05d_00*s' % test_id)]
                    gt_path = gt_files[0]
                    self.test_data_filenames.append([in_path1, in_path2, gt_path])
        print(len(self.test_data_filenames))

        def gen():
            for i in range(len(self.test_data_filenames)):
                in_path1, in_path2, gt_path = self.test_data_filenames[i]
                gt_rgb = self.long_data[gt_path][:]
                gt_edge = self.long_data[gt_path + ".edge"][:]
                num_positive = np.count_nonzero(gt_edge == 1)
                num_negative = np.count_nonzero(gt_edge == 0)
                gt_mask = np.float32(gt_edge)
                gt_mask[gt_mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
                gt_mask[gt_mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
                yield (gt_rgb, gt_mask, gt_edge)

        def map(gt_rgb, gt_mask, gt_edge):
            # normalization
            gt_rgb = tf.cast(gt_rgb, tf.float32)
            gt_rgb = gt_rgb / 65535.0
            gt_edge = tf.cast(gt_edge, tf.float32)

            # reshape
            gt_rgb.set_shape([self.H, self.W, 3])
            gt_rgb = gt_rgb[0:self.H_out, 0:self.W_out, :]
            gt_rgb.set_shape([self.H_out, self.W_out, 3])

            gt_edge.set_shape([self.H, self.W, 1])
            gt_edge = gt_edge[0:self.H_out, 0:self.W_out, :]
            gt_edge.set_shape([self.H_out, self.W_out, 1])

            # gt_mask = gt_mask.set_shape([4032, 6032, 1])
            gt_mask = gt_mask[0:self.H_out, 0:self.W_out, :]
            # gt_mask = gt_mask.set_shape([self.H_out, self.W_out, 1])

            feature = {'image': gt_rgb, 'mask': gt_mask}  # 'in_rgb': in_rgb,
            return feature, gt_edge

        dataset = tf.data.Dataset.from_generator(gen, (tf.uint16, tf.float32, tf.bool))
        #dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1))
        dataset = dataset.map(map_func=map, num_parallel_calls=8)
        dataset = dataset.batch(batch_size=1, drop_remainder=True)
        dataset = dataset.prefetch(10)
        return dataset

def main():
    data = SID_dataset(set_name="Sony", debug=False)
    data.save_short_data_as_np_file()
    data.save_long_data_as_np_file()

    data = SID_dataset(set_name="Fuji", debug=False)
    data.save_short_data_as_np_file()
    data.save_long_data_as_np_file()

if __name__ == '__main__':
    main()


