import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets

def resblock(x, filter_size):
    res = slim.conv2d(x, filter_size, [3, 3], activation_fn=lrelu, padding='SAME')
    res = slim.conv2d(res, filter_size, [3, 3], activation_fn=lrelu, padding='SAME')
    return x + res

def lrelu(x):
    return tf.maximum(x * 0.2, x)

def upsample_and_concat(x1, x2, output_channels, in_channels, scope):
    deconv_output = slim.conv2d_transpose(x1, output_channels, [2, 2], stride=2, padding='SAME',
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.02), scope=scope)
    H, W = tf.shape(deconv_output)[1], tf.shape(deconv_output)[2]
    h, w = tf.shape(x2)[1], tf.shape(x2)[2]
    H = tf.cast(H, tf.float32)
    W = tf.cast(W, tf.float32)
    h = tf.cast(h, tf.float32)
    w = tf.cast(w, tf.float32)
    x1 = tf.cast(tf.round((W - w) / 2), tf.int32)
    y1 = tf.cast(tf.round((H - h) / 2), tf.int32)
    h = tf.cast(h, tf.int32)
    w = tf.cast(w, tf.int32)
    deconv_output = deconv_output[:, y1: y1 + h, x1: x1 + w, :]
    deconv_output = tf.concat([deconv_output, x2], 3)
    #deconv_output.set_shape([None, None, None, output_channels * 2])
    return deconv_output

class Netowrk():
    def __init__(self, set_name):
        self.set_name = set_name

    def SID(self, input):
        # def upsample_and_concat_SID(x1, x2, output_channels, in_channels, scope):
        #     pool_size = 2
        #     deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
        #     deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])
        #
        #     deconv_output = tf.concat([deconv, x2], 3)
        #     deconv_output.set_shape([None, None, None, output_channels * 2])
        #
        #     return deconv_output

        with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
            conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
            conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
            pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

            conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
            conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
            pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

            conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
            conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
            pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

            conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
            conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
            pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

            conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
            conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

            up6 = upsample_and_concat(conv5, conv4, 256, 512, scope='up6')
            conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
            conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

            up7 = upsample_and_concat(conv6, conv3, 128, 256, scope='up7')
            conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
            conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

            up8 = upsample_and_concat(conv7, conv2, 64, 128, scope='up8')
            conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
            conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

            up9 = upsample_and_concat(conv8, conv1, 32, 64, scope='up9')
            conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
            conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

            if self.set_name == 'Sony':
                conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='conv10')
                out = tf.depth_to_space(conv10, 2)
            elif self.set_name == 'Fuji':
                conv10 = slim.conv2d(conv9, 27, [1, 1], rate=1, activation_fn=None, scope='conv10')
                out = tf.depth_to_space(conv10, 3)
        return out


    def HDR_concat(self, input):
        with tf.variable_scope("hdr", reuse=tf.AUTO_REUSE):
            conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
            conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
            pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

            conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
            conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
            pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

            conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
            conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
            pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

            conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
            conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
            pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

            conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
            conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

            up6 = upsample_and_concat(conv5, conv4, 256, 512, scope='up6')
            conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
            conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

            up7 = upsample_and_concat(conv6, conv3, 128, 256, scope='up7')
            conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
            conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

            up8 = upsample_and_concat(conv7, conv2, 64, 128, scope='up8')
            conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
            conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

            up9 = upsample_and_concat(conv8, conv1, 32, 64, scope='up9')
            conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
            conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

            if self.set_name == 'Sony':
                conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='conv10')
                out = tf.depth_to_space(conv10, 2)
            elif self.set_name == 'Fuji':
                conv10 = slim.conv2d(conv9, 27, [1, 1], rate=1, activation_fn=None, scope='conv10')
                out = tf.depth_to_space(conv10, 3)
        return out

    def HDR_fusion_old(self, x1, x2):
        def exchange_block(f1, f2, out_channels, scope):
            max_ = tf.maximum(f1, f2)
            avg_ = (f1 + f2) / 2
            out_ = tf.concat([max_, avg_], 3)
            out_ = slim.conv2d(out_, out_channels, [1, 1], activation_fn=None, biases_initializer=None, scope=scope)
            return out_

        filter_size = 32
        with tf.variable_scope("hdr_fusion", reuse=tf.AUTO_REUSE):
            conv1_in1 = slim.conv2d(x1, filter_size, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv1_1')
            conv1_in1 = slim.conv2d(conv1_in1, filter_size, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv1_2')
            conv1_in2 = slim.conv2d(x2, filter_size, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv1_1')
            conv1_in2 = slim.conv2d(conv1_in2, filter_size, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv1_2')
            pool1 = exchange_block(conv1_in1, conv1_in2, filter_size, scope="ex1")
            pool1_in1 = slim.max_pool2d(tf.concat([conv1_in1, pool1], 3), [2, 2], padding='SAME')
            pool1_in2 = slim.max_pool2d(tf.concat([conv1_in2, pool1], 3), [2, 2], padding='SAME')

            conv2_in1 = slim.conv2d(pool1_in1, filter_size * 2, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv2_1')
            conv2_in1 = slim.conv2d(conv2_in1, filter_size * 2, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv2_2')
            conv2_in2 = slim.conv2d(pool1_in2, filter_size * 2, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv2_1')
            conv2_in2 = slim.conv2d(conv2_in2, filter_size * 2, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv2_2')
            pool2 = exchange_block(conv2_in1, conv2_in2, filter_size*2, scope="ex2")
            pool2_in1 = slim.max_pool2d(tf.concat([conv2_in1, pool2], 3), [2, 2], padding='SAME')
            pool2_in2 = slim.max_pool2d(tf.concat([conv2_in2, pool2], 3), [2, 2], padding='SAME')

            conv3_in1 = slim.conv2d(pool2_in1, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv3_1')
            conv3_in1 = slim.conv2d(conv3_in1, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv3_2')
            conv3_in1 = slim.conv2d(conv3_in1, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv3_3')
            conv3_in2 = slim.conv2d(pool2_in2, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv3_1')
            conv3_in2 = slim.conv2d(conv3_in2, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv3_2')
            conv3_in2 = slim.conv2d(conv3_in2, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv3_3')
            pool3 = exchange_block(conv3_in1, conv3_in2, filter_size*4, scope="ex3")
            pool3_in1 = slim.max_pool2d(tf.concat([conv3_in1, pool3], 3), [2, 2], padding='SAME')
            pool3_in2 = slim.max_pool2d(tf.concat([conv3_in2, pool3], 3), [2, 2], padding='SAME')

            conv4_in1 = slim.conv2d(pool3_in1, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv4_1')
            conv4_in1 = slim.conv2d(conv4_in1, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv4_2')
            conv4_in1 = slim.conv2d(conv4_in1, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv4_3')
            conv4_in2 = slim.conv2d(pool3_in2, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv4_1')
            conv4_in2 = slim.conv2d(conv4_in2, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv4_2')
            conv4_in2 = slim.conv2d(conv4_in2, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv4_3')
            pool4 = exchange_block(conv4_in1, conv4_in2, filter_size*8, scope="ex4")
            pool4_in1 = slim.max_pool2d(tf.concat([conv4_in1, pool4], 3), [2, 2], padding='SAME')
            pool4_in2 = slim.max_pool2d(tf.concat([conv4_in2, pool4], 3), [2, 2], padding='SAME')

            conv5_in1 = slim.conv2d(pool4_in1, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv5_1')
            conv5_in1 = slim.conv2d(conv5_in1, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv5_2')
            conv5_in1 = slim.conv2d(conv5_in1, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv5_3')
            conv5_in2 = slim.conv2d(pool4_in2, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv5_1')
            conv5_in2 = slim.conv2d(conv5_in2, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv5_2')
            conv5_in2 = slim.conv2d(conv5_in2, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv5_3')

            up6_in1 = upsample_and_concat(conv5_in1, conv4_in1, filter_size * 8, filter_size * 16, 'b1_up6')
            up6_in2 = upsample_and_concat(conv5_in2, conv4_in2, filter_size * 8, filter_size * 16, 'b2_up6')
            up6 = exchange_block(up6_in1, up6_in2, filter_size*8, scope="ex6")
            up6_in1_ = tf.concat([up6_in1, up6], 3)
            up6_in2_ = tf.concat([up6_in2, up6], 3)
            conv6_in1 = slim.conv2d(up6_in1_, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv6_1')
            conv6_in1 = slim.conv2d(conv6_in1, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv6_2')
            conv6_in1 = slim.conv2d(conv6_in1, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv6_3')
            conv6_in2 = slim.conv2d(up6_in2_, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv6_1')
            conv6_in2 = slim.conv2d(conv6_in2, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv6_2')
            conv6_in2 = slim.conv2d(conv6_in2, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv6_3')

            up7_in1 = upsample_and_concat(conv6_in1, conv3_in1, filter_size * 4, filter_size * 8, 'b1_up7')
            up7_in2 = upsample_and_concat(conv6_in2, conv3_in2, filter_size * 4, filter_size * 8, 'b2_up7')
            up7 = exchange_block(up7_in1, up7_in2, filter_size*4, scope="ex7")
            up7_in1_ = tf.concat([up7_in1, up7], 3)
            up7_in2_ = tf.concat([up7_in2, up7], 3)
            conv7_in1 = slim.conv2d(up7_in1_, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv7_1')
            conv7_in1 = slim.conv2d(conv7_in1, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv7_2')
            conv7_in1 = slim.conv2d(conv7_in1, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv7_3')
            conv7_in2 = slim.conv2d(up7_in2_, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv7_1')
            conv7_in2 = slim.conv2d(conv7_in2, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv7_2')
            conv7_in2 = slim.conv2d(conv7_in2, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv7_3')

            up8_in1 = upsample_and_concat(conv7_in1, conv2_in1, filter_size * 2, filter_size * 4, 'b1_up8')
            up8_in2 = upsample_and_concat(conv7_in2, conv2_in2, filter_size * 2, filter_size * 4, 'b2_up8')
            up8 = exchange_block(up8_in1, up8_in2, filter_size*2, scope="ex8")
            up8_in1_ = tf.concat([up8_in1, up8], 3)
            up8_in2_ = tf.concat([up8_in2, up8], 3)
            conv8_in1 = slim.conv2d(up8_in1_, filter_size * 2, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv8_1')
            conv8_in1 = slim.conv2d(conv8_in1, filter_size * 2, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv8_2')
            conv8_in2 = slim.conv2d(up8_in2_, filter_size * 2, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv8_1')
            conv8_in2 = slim.conv2d(conv8_in2, filter_size * 2, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv8_2')

            up9_in1 = upsample_and_concat(conv8_in1, conv1_in1, filter_size, filter_size * 2, 'b1_up9')
            up9_in2 = upsample_and_concat(conv8_in2, conv1_in2, filter_size, filter_size * 2, 'b2_up9')
            up9 = exchange_block(up9_in1, up9_in2, filter_size, scope="ex9")
            conv9_in1 = slim.conv2d(tf.concat([up9_in1, up9], 3), filter_size, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv9_1')
            conv9_in1 = slim.conv2d(conv9_in1, filter_size, [3, 3], rate=1, activation_fn=lrelu, scope='b1_conv9_2')
            conv9_in2 = slim.conv2d(tf.concat([up9_in2, up9], 3), filter_size, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv9_1')
            conv9_in2 = slim.conv2d(conv9_in2, filter_size, [3, 3], rate=1, activation_fn=lrelu, scope='b2_conv9_2')
            conv9 = tf.concat([conv9_in1, conv9_in2], 3)

            if self.set_name == 'Sony':
                conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='conv10')
                out = tf.depth_to_space(conv10, 2)
            elif self.set_name == 'Fuji':
                conv10 = slim.conv2d(conv9, 27, [1, 1], rate=1, activation_fn=None, scope='conv10')
                out = tf.depth_to_space(conv10, 3)

            return out, conv10, conv9

    def HDR_fusion(self, x1, x2):
        def exchange_block(f1, f2, out_channels, scope):
            max_ = tf.maximum(f1, f2)
            avg_ = (f1 + f2) / 2
            out_ = tf.concat([max_, avg_], 3)
            out_ = slim.conv2d(out_, out_channels, [1, 1], activation_fn=None, biases_initializer=None, scope=scope)
            return out_

        def conv_layer(x, filter_size, n=2, scope='conv'):
            conv = slim.conv2d(x, filter_size, [3, 3], activation_fn=lrelu, scope=scope+'_1')
            conv = slim.conv2d(conv, filter_size, [3, 3], activation_fn=lrelu, scope=scope+'_2')
            if n==3:
                conv = slim.conv2d(conv, filter_size, [3, 3], activation_fn=lrelu, scope=scope + '_3')
            return conv

        def down_block(x1, x2, filter_size, scope='1'):
            if scope in ['3', '4']:
                layer_n=3
            else:
                layer_n=2
            conv_in1 = conv_layer(x1, filter_size, layer_n, scope='b1_conv'+scope)
            conv_in2 = conv_layer(x2, filter_size, layer_n, scope='b2_conv'+scope)
            pool = exchange_block(conv_in1, conv_in2, filter_size, scope="ex"+scope)
            pool_in1 = slim.max_pool2d(tf.concat([conv_in1, pool], 3), [2, 2], padding='SAME')
            pool_in2 = slim.max_pool2d(tf.concat([conv_in2, pool], 3), [2, 2], padding='SAME')
            return conv_in1, conv_in2, pool_in1, pool_in2

        def up_block(x1, x2, conv1, conv2, filter_size, scope='1'):
            up_in1 = upsample_and_concat(x1, conv1, filter_size, filter_size * 2, 'b1_up'+scope)
            up_in2 = upsample_and_concat(x2, conv2, filter_size, filter_size * 2, 'b2_up'+scope)
            up = exchange_block(up_in1, up_in2, filter_size, scope="ex"+scope)
            up_in1_ = tf.concat([up_in1, up], 3)
            up_in2_ = tf.concat([up_in2, up], 3)
            if scope in ['6', '7']:
                layer_n=3
            else:
                layer_n=2
            x1 = conv_layer(up_in1_, filter_size, layer_n, scope='b1_conv'+scope)
            x2 = conv_layer(up_in2_, filter_size, layer_n, scope='b2_conv'+scope)
            return x1, x2

        filter_size = 32
        with tf.variable_scope("hdr_fusion", reuse=tf.AUTO_REUSE):
            conv1_in1, conv1_in2, x1, x2 = down_block(x1, x2, filter_size, scope='1')

            conv2_in1, conv2_in2, x1, x2 = down_block(x1, x2, filter_size * 2, scope='2')

            conv3_in1, conv3_in2, x1, x2 = down_block(x1, x2, filter_size * 4, scope='3')

            conv4_in1, conv4_in2, x1, x2 = down_block(x1, x2, filter_size * 8, scope='4')

            x1 = conv_layer(x1, filter_size * 8, 3, scope='b1_conv5')
            x2 = conv_layer(x2, filter_size * 8, 3, scope='b2_conv5')

            x1, x2 = up_block(x1, x2, conv4_in1, conv4_in2, filter_size * 8, scope='6')

            x1, x2 = up_block(x1, x2, conv3_in1, conv3_in2, filter_size * 4, scope='7')

            x1, x2 = up_block(x1, x2, conv2_in1, conv2_in2, filter_size * 2, scope='8')

            x1, x2 = up_block(x1, x2, conv1_in1, conv1_in2, filter_size, scope='9')

            x = tf.concat([x1, x2], 3)

            if self.set_name == 'Sony':
                conv10 = slim.conv2d(x, 12, [1, 1], activation_fn=None, scope='conv10')
                out = tf.depth_to_space(conv10, 2)
            elif self.set_name == 'Fuji':
                conv10 = slim.conv2d(x, 27, [1, 1], activation_fn=None, scope='conv10')
                out = tf.depth_to_space(conv10, 3)

            return out, conv10, x

    def HDR_fusion_3inputs(self, x1, x2, x3):

        def exchange_block(f1, f2, f3, out_channels, scope):
            max_ = tf.maximum(f1, f2)
            max_ = tf.maximum(max_, f3)
            avg_ = (f1 + f2 + f3) / 3
            out_ = tf.concat([max_, avg_], 3)
            out_ = slim.conv2d(out_, out_channels, [1, 1], activation_fn=None, biases_initializer=None, scope=scope)
            return out_

        def conv_layer(x, filter_size, n=2, scope='conv'):
            conv = slim.conv2d(x, filter_size, [3, 3], activation_fn=lrelu, scope=scope+'_1')
            conv = slim.conv2d(conv, filter_size, [3, 3], activation_fn=lrelu, scope=scope+'_2')
            if n==3:
                conv = slim.conv2d(conv, filter_size, [3, 3], activation_fn=lrelu, scope=scope + '_3')
            return conv

        def down_block(x1, x2, x3, filter_size, scope='1'):
            if scope in ['3', '4']:
                layer_n=3
            else:
                layer_n=2
            conv_in1 = conv_layer(x1, filter_size, layer_n, scope='b1_conv'+scope)
            conv_in2 = conv_layer(x2, filter_size, layer_n, scope='b2_conv'+scope)
            conv_in3 = conv_layer(x3, filter_size, layer_n, scope='b3_conv'+scope)
            pool = exchange_block(conv_in1, conv_in2, conv_in3, filter_size, scope="ex"+scope)
            pool_in1 = slim.max_pool2d(tf.concat([conv_in1, pool], 3), [2, 2], padding='SAME')
            pool_in2 = slim.max_pool2d(tf.concat([conv_in2, pool], 3), [2, 2], padding='SAME')
            pool_in3 = slim.max_pool2d(tf.concat([conv_in3, pool], 3), [2, 2], padding='SAME')
            return conv_in1, conv_in2, conv_in3, pool_in1, pool_in2, pool_in3

        def up_block(x1, x2, x3, conv1, conv2, conv3, filter_size, scope='1'):
            up_in1 = upsample_and_concat(x1, conv1, filter_size, filter_size * 2, 'b1_up'+scope)
            up_in2 = upsample_and_concat(x2, conv2, filter_size, filter_size * 2, 'b2_up'+scope)
            up_in3 = upsample_and_concat(x3, conv3, filter_size, filter_size * 2, 'b3_up'+scope)
            up = exchange_block(up_in1, up_in2, up_in3, filter_size, scope="ex"+scope)
            up_in1_ = tf.concat([up_in1, up], 3)
            up_in2_ = tf.concat([up_in2, up], 3)
            up_in3_ = tf.concat([up_in3, up], 3)
            if scope in ['6', '7']:
                layer_n=3
            else:
                layer_n=2
            x1 = conv_layer(up_in1_, filter_size, layer_n, scope='b1_conv'+scope)
            x2 = conv_layer(up_in2_, filter_size, layer_n, scope='b2_conv'+scope)
            x3 = conv_layer(up_in3_, filter_size, layer_n, scope='b3_conv'+scope)
            return x1, x2, x3


        filter_size = 32
        with tf.variable_scope("hdr_fusion", reuse=tf.AUTO_REUSE):
            conv1_in1, conv1_in2, conv1_in3, x1, x2, x3 = down_block(x1, x2, x3, filter_size, scope='1')

            conv2_in1, conv2_in2, conv2_in3, x1, x2, x3 = down_block(x1, x2, x3, filter_size * 2, scope='2')

            conv3_in1, conv3_in2, conv3_in3, x1, x2, x3 = down_block(x1, x2, x3, filter_size * 4, scope='3')

            conv4_in1, conv4_in2, conv4_in3, x1, x2, x3 = down_block(x1, x2, x3, filter_size * 8, scope='4')

            x1 = conv_layer(x1, filter_size * 8, 3, scope='b1_conv5')
            x2 = conv_layer(x2, filter_size * 8, 3, scope='b2_conv5')
            x3 = conv_layer(x3, filter_size * 8, 3, scope='b3_conv5')

            x1, x2, x3 = up_block(x1, x2, x3, conv4_in1, conv4_in2, conv4_in3, filter_size * 8, scope='6')

            x1, x2, x3 = up_block(x1, x2, x3, conv3_in1, conv3_in2, conv3_in3, filter_size * 4, scope='7')

            x1, x2, x3 = up_block(x1, x2, x3, conv2_in1, conv2_in2, conv2_in3, filter_size * 2, scope='8')

            x1, x2, x3 = up_block(x1, x2, x3, conv1_in1, conv1_in2, conv1_in3, filter_size, scope='9')

            x = tf.concat([x1, x2, x3], 3)

            if self.set_name == 'Sony':
                conv10 = slim.conv2d(x, 12, [1, 1], activation_fn=None, scope='conv10')
                out = tf.depth_to_space(conv10, 2)
            elif self.set_name == 'Fuji':
                conv10 = slim.conv2d(x, 27, [1, 1], activation_fn=None, scope='conv10')
                out = tf.depth_to_space(conv10, 3)

            return out, conv10, x

    def edge(self, input):
        def crop_center(x, h, w):
            H, W = tf.shape(x)[1], tf.shape(x)[2]
            H = tf.cast(H, tf.float32)
            W = tf.cast(W, tf.float32)
            h = tf.cast(h, tf.float32)
            w = tf.cast(w, tf.float32)
            x1 = tf.cast(tf.round((W - w) / 2), tf.int32)
            y1 = tf.cast(tf.round((H - h) / 2), tf.int32)
            h = tf.cast(h, tf.int32)
            w = tf.cast(w, tf.int32)
            x = x[:, y1: y1 + h, x1: x1 + w, :]
            return x
        filter_size = 64
        filter_size2 = 32
        if self.set_name == 'Sony':
            out_channel = 4
        if self.set_name == 'Fuji':
            out_channel = 9
        with tf.variable_scope("edge", reuse=tf.AUTO_REUSE):
            # stage 1
            conv1_1 = slim.conv2d(input, filter_size, [3, 3], activation_fn=tf.nn.relu, scope='conv1_1')
            conv1_2 = slim.conv2d(conv1_1, filter_size, [3, 3], activation_fn=tf.nn.relu, scope='conv1_2')
            pool1 = slim.max_pool2d(conv1_2, [2, 2], padding='SAME')
            conv1_1_down = slim.conv2d(conv1_1, filter_size2, [1, 1], activation_fn=None, scope='conv1_1_down')
            conv1_2_down = slim.conv2d(conv1_2, filter_size2, [1, 1], activation_fn=None, scope='conv1_2_down')
            s1_out = slim.conv2d(conv1_1_down + conv1_2_down, out_channel, [1, 1], activation_fn=None, scope='conv_1')

            # stage 2
            conv2_1 = slim.conv2d(pool1, filter_size * 2, [3, 3], activation_fn=tf.nn.relu, scope='conv2_1')
            conv2_2 = slim.conv2d(conv2_1, filter_size * 2, [3, 3], activation_fn=tf.nn.relu, scope='conv2_2')
            pool2 = slim.max_pool2d(conv2_2, [2, 2], padding='SAME')
            conv2_1_down = slim.conv2d(conv2_1, filter_size2, [1, 1], activation_fn=None, scope='conv2_1_down')
            conv2_2_down = slim.conv2d(conv2_2, filter_size2, [1, 1], activation_fn=None, scope='conv2_2_down')
            s2_out = slim.conv2d(conv2_1_down + conv2_2_down, out_channel, [1, 1], activation_fn=None, scope='conv_2')
            s2_out = slim.conv2d_transpose(s2_out, out_channel, [4, 4], stride=2, padding='SAME', activation_fn=None, scope='conv_t_2')

            # stage 3
            conv3_1 = slim.conv2d(pool2, filter_size * 4, [3, 3], activation_fn=tf.nn.relu, scope='conv3_1')
            conv3_2 = slim.conv2d(conv3_1, filter_size * 4, [3, 3], activation_fn=tf.nn.relu, scope='conv3_2')
            conv3_3 = slim.conv2d(conv3_2, filter_size * 4, [3, 3], activation_fn=tf.nn.relu, scope='conv3_3')
            pool3 = slim.max_pool2d(conv3_3, [2, 2], padding='SAME')
            conv3_1_down = slim.conv2d(conv3_1, filter_size2, [1, 1], activation_fn=None, scope='conv3_1_down')
            conv3_2_down = slim.conv2d(conv3_2, filter_size2, [1, 1], activation_fn=None, scope='conv3_2_down')
            conv3_3_down = slim.conv2d(conv3_3, filter_size2, [1, 1], activation_fn=None, scope='conv3_3_down')
            s3_out = slim.conv2d(conv3_1_down + conv3_2_down + conv3_3_down, out_channel, [1, 1], activation_fn=None, scope='conv_3')
            s3_out = slim.conv2d_transpose(s3_out, out_channel, [8, 8], stride=4, padding='SAME', activation_fn=None, scope='conv_t_3')

            # stage 4
            conv4_1 = slim.conv2d(pool3, filter_size * 8, [3, 3], activation_fn=tf.nn.relu, scope='conv4_1')
            conv4_2 = slim.conv2d(conv4_1, filter_size * 8, [3, 3], activation_fn=tf.nn.relu, scope='conv4_2')
            conv4_3 = slim.conv2d(conv4_2, filter_size * 8, [3, 3], activation_fn=tf.nn.relu, scope='conv4_3')
            pool4 = slim.max_pool2d(conv4_3, [2, 2], stride=1, padding='SAME')
            conv4_1_down = slim.conv2d(conv4_1, filter_size2, [1, 1], activation_fn=None, scope='conv4_1_down')
            conv4_2_down = slim.conv2d(conv4_2, filter_size2, [1, 1], activation_fn=None, scope='conv4_2_down')
            conv4_3_down = slim.conv2d(conv4_3, filter_size2, [1, 1], activation_fn=None, scope='conv4_3_down')
            s4_out = slim.conv2d(conv4_1_down + conv4_2_down + conv4_3_down, out_channel, [1, 1], activation_fn=None, scope='conv_4')
            s4_out = slim.conv2d_transpose(s4_out, out_channel, [16, 16], stride=8, padding='SAME', activation_fn=None, scope='conv_t_4')

            conv5_1 = slim.conv2d(pool4, filter_size * 8, [3, 3], activation_fn=tf.nn.relu, scope='conv5_1')
            conv5_2 = slim.conv2d(conv5_1, filter_size * 8, [3, 3], activation_fn=tf.nn.relu, scope='conv5_2')
            conv5_3 = slim.conv2d(conv5_2, filter_size * 8, [3, 3], activation_fn=tf.nn.relu, scope='conv5_3')
            conv5_1_down = slim.conv2d(conv5_1, filter_size2, [1, 1], activation_fn=None, scope='conv5_1_down')
            conv5_2_down = slim.conv2d(conv5_2, filter_size2, [1, 1], activation_fn=None, scope='conv5_2_down')
            conv5_3_down = slim.conv2d(conv5_3, filter_size2, [1, 1], activation_fn=None, scope='conv5_3_down')
            s5_out = slim.conv2d(conv5_1_down + conv5_2_down + conv5_3_down, out_channel, [1, 1], activation_fn=None, scope='conv_5')
            s5_out = slim.conv2d_transpose(s5_out, out_channel, [32, 32], stride=8, padding='SAME', activation_fn=None, scope='conv_t_5')

            h, w = tf.shape(input)[1], tf.shape(input)[2]
            s2_out = crop_center(s2_out, h, w)
            s3_out = crop_center(s3_out, h, w)
            s4_out = crop_center(s4_out, h, w)
            s5_out = crop_center(s5_out, h, w)
            fuse = tf.concat([s1_out, s2_out, s3_out, s4_out, s5_out], 3)
            fuse_out = slim.conv2d(fuse, out_channel, [1, 1], activation_fn=None, scope='conv_fuse')

            return [s1_out, s2_out, s3_out, s4_out, s5_out, fuse_out]

    def edge_loss(self, prediction, label, mask):
        return tf.losses.sigmoid_cross_entropy(label, prediction, mask)


    def merge(self, input, edge, feature):
        input = tf.concat([input, edge, feature], 3)
        filter_size = 32
        with tf.variable_scope("merge", reuse=tf.AUTO_REUSE):
            conv1 = slim.conv2d(input, filter_size * 2, [3, 3], rate=1, activation_fn=lrelu, scope='conv1_1')
            conv1 = slim.conv2d(conv1, filter_size * 2, [3, 3], rate=1, activation_fn=lrelu, scope='conv1_2')
            pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

            conv2 = slim.conv2d(pool1, filter_size * 2, [3, 3], rate=1, activation_fn=lrelu, scope='conv2_1')
            conv2 = slim.conv2d(conv2, filter_size * 2, [3, 3], rate=1, activation_fn=lrelu, scope='conv2_2')
            pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

            conv3 = slim.conv2d(pool2, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='conv3_1')
            conv3 = slim.conv2d(conv3, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='conv3_2')
            conv3 = slim.conv2d(conv3, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='conv3_3')
            pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

            conv4 = slim.conv2d(pool3, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='conv4_1')
            conv4 = slim.conv2d(conv4, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='conv4_2')
            conv4 = slim.conv2d(conv4, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='conv4_3')
            pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

            conv5 = slim.conv2d(pool4, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='conv5_1')
            conv5 = slim.conv2d(conv5, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='conv5_2')
            conv5 = slim.conv2d(conv5, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='conv5_3')

            up6 = upsample_and_concat(conv5, conv4, filter_size * 8, filter_size * 16, 'up6')
            conv6 = slim.conv2d(up6, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='conv6_1')
            conv6 = slim.conv2d(conv6, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='conv6_2')
            conv6 = slim.conv2d(conv6, filter_size * 8, [3, 3], rate=1, activation_fn=lrelu, scope='conv6_3')

            up7 = upsample_and_concat(conv6, conv3, filter_size * 4, filter_size * 8, 'up7')
            conv7 = slim.conv2d(up7, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='conv7_1')
            conv7 = slim.conv2d(conv7, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='conv7_2')
            conv7 = slim.conv2d(conv7, filter_size * 4, [3, 3], rate=1, activation_fn=lrelu, scope='conv7_3')

            up8 = upsample_and_concat(conv7, conv2, filter_size * 2, filter_size * 4, 'up8')
            conv8 = slim.conv2d(up8, filter_size * 2, [3, 3], rate=1, activation_fn=lrelu, scope='conv8_1')
            conv8 = slim.conv2d(conv8, filter_size * 2, [3, 3], rate=1, activation_fn=lrelu, scope='conv8_2')

            up9 = upsample_and_concat(conv8, conv1, filter_size*2, filter_size * 2, 'up9')
            conv9 = slim.conv2d(up9, filter_size*2, [3, 3], rate=1, activation_fn=lrelu, scope='conv9_1')
            conv9 = slim.conv2d(conv9, filter_size*2, [3, 3], rate=1, activation_fn=lrelu, scope='conv9_2')

            conv9 = tf.concat([conv9, feature, edge], 3)

            if self.set_name == 'Sony':
                conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='conv10')
                out = tf.depth_to_space(conv10, 2)
            elif self.set_name == 'Fuji':
                conv10 = slim.conv2d(conv9, 27, [1, 1], rate=1, activation_fn=None, scope='conv10')
                out = tf.depth_to_space(conv10, 3)
        return out

    def main(self, input1, input2):
        hdr_out, out_, feature_ = self.HDR_fusion(input1, input2)
        # edge_wo_sigmoid = self.edge(out_)
        [gen_edge1, gen_edge2, gen_edge3, gen_edge4, gen_edge5, edge_wo_sigmoid] = self.edge(out_)
        edge = tf.sigmoid(edge_wo_sigmoid)
        # feature = tf.concat([feature_, edge, input1, input2], 3)
        input = tf.concat([input1, input2], 3)
        out = self.merge(input, edge, feature_)
        return out, edge, hdr_out, [gen_edge1, gen_edge2, gen_edge3, gen_edge4, gen_edge5, edge_wo_sigmoid]

    def main_edge(self, input1, input2):
        hdr_out, out_, feature_ = self.HDR_fusion(input1, input2)
        [gen_edge1, gen_edge2, gen_edge3, gen_edge4, gen_edge5, edge_wo_sigmoid] = self.edge(out_)
        edge = tf.sigmoid(edge_wo_sigmoid)
        return hdr_out, edge, hdr_out, [edge_wo_sigmoid]

    def loss_model(self, x):
        x = x * 2 - 1
        logits, endpoints_dict = slim.nets.vgg.vgg_16(x, spatial_squeeze=False)
        return logits, endpoints_dict



    def main3(self, input1, input2, input3):
        hdr_out, out_, feature_ = self.HDR_fusion_3inputs(input1, input2, input3)
        # edge_wo_sigmoid = self.edge(out_)
        [gen_edge1, gen_edge2, gen_edge3, gen_edge4, gen_edge5, edge_wo_sigmoid] = self.edge(out_)
        edge = tf.sigmoid(edge_wo_sigmoid)
        input = tf.concat([input1, input2, input3], 3)
        out = self.merge(input, edge, feature_)
        return out, edge, hdr_out, [gen_edge1, gen_edge2, gen_edge3, gen_edge4, gen_edge5, edge_wo_sigmoid]
