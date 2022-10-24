from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.platform import flags
from tensorflow.python.platform import app
from tensorflow.python import pywrap_tensorflow
import numpy as np
import tensorflow as tf
import tflearn
import re
import sys
from keras import backend as K
from keras.layers import GlobalAveragePooling3D, GlobalMaxPooling3D, Reshape, Dense, Add, Activation

def cbam_block(input_feature, name, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    with tf.variable_scope(name):
        attention_feature = channel_attention(input_feature, 'ch_at', ratio)
        attention_feature = spatial_attention(attention_feature, 'sp_at')
    return attention_feature

'''def channel_attention(input_feature, name, ratio=8):    
    with tf.variable_scope(name):
        
        channel = input_feature.get_shape()[-1]
        avg_pool = tf.reduce_mean(input_feature, axis=[1,2,3], keepdims=True)

        avg_pool = tflearn.fully_connected(incoming = avg_pool,n_units = channel//ratio,activation = 'relu',name='mlp_0',scope = 'mlp_0',reuse=None)
        avg_pool = tflearn.fully_connected(incoming = avg_pool,n_units = channel,name='mlp_1',scope = 'mlp_1',reuse=None)

        max_pool = tf.reduce_max(input_feature, axis=[1,2,3], keepdims=True)
        max_pool = tflearn.fully_connected(incoming = max_pool,n_units = channel//ratio,activation = 'relu',name='mlp_0',scope = 'mlp_0',reuse=True)
        max_pool = tflearn.fully_connected(incoming = max_pool,n_units = channel,name='mlp_1',scope = 'mlp_1',reuse=True)

        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')
        
    return input_feature * scale'''

def channel_attention(input_feature, ratio=8):
        #channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = input_feature.shape.as_list()[-1]

        shared_layer_one = Dense(channel//ratio,
                                                         activation='relu',
                                                         kernel_initializer='he_normal',
                                                         use_bias=True,
                                                         bias_initializer='zeros')
        shared_layer_two = Dense(channel,
                                                         kernel_initializer='he_normal',
                                                         use_bias=True,
                                                         bias_initializer='zeros')

        avg_pool = GlobalAveragePooling3D()(input_feature)
        avg_pool = Reshape((1,1,1,channel))(avg_pool)
        #print(avg_pool.shape.as_list()[1:],channel)
        #assert avg_pool.shape.as_list()[1:] == (1,1,1,channel)
        avg_pool = shared_layer_one(avg_pool)
        #assert avg_pool.shape.as_list()[1:] == (1,1,1,channel//ratio)
        avg_pool = shared_layer_two(avg_pool)
        #assert avg_pool.shape.as_list()[1:] == (1,1,1,channel)

        max_pool = GlobalMaxPooling3D()(input_feature)
        max_pool = Reshape((1,1,1,channel))(max_pool)
        #assert max_pool.shape.as_list()[1:] == (1,1,1,channel)
        max_pool = shared_layer_one(max_pool)
        #assert max_pool.shape.as_list()[1:] == (1,1,1,channel//ratio)
        max_pool = shared_layer_two(max_pool)
        #assert max_pool.shape.as_list()[1:] == (1,1,1,channel)

        cbam_feature = Add()([avg_pool,max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)

        return cbam_feature



def spatial_attention(input_feature, name):
    kernel_size = 7
    with tf.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[4], keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[4], keepdims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool,max_pool], 4)
        assert concat.get_shape()[-1] == 2
        
        '''concat = tf.layers.conv2d(concat,
                                filters=1,
                                kernel_size=[kernel_size,kernel_size],
                                strides=[1,1],
                                padding="same",
                                activation=None,
                                kernel_initializer=kernel_initializer,
                                use_bias=False,
                                name='conv')
        '''

        concat = tflearn.layers.conv_3d(concat, 1, kernel_size, strides=1,
                                  padding='same', activation='linear', bias=True, scope='conv3d_spatial_AT', reuse=False, weights_init='uniform_scaling')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')
        
    return input_feature * concat


def ReLU(target, name=None):
    return tflearn.activations.relu(target)


def LeakyReLU(target, alpha=0.1, name=None):
    return tflearn.activations.leaky_relu(target, alpha=alpha, name=name)

def Softmax(target,  name=None):
    return tflearn.activations.softmax(target)

def Sigmoid(target,  name=None):
    return tflearn.activations.sigmoid(target)

def convolve(opName, inputLayer, inputChannel, outputChannel, kernelSize, stride, stddev=1e-2):
    return tflearn.layers.conv_2d(inputLayer, outputChannel, kernelSize, strides=stride,
                                  padding='same', activation='linear', bias=True, scope=opName)
    # kernelVariables = tf.Variable(
    #     tf.truncated_normal(
    #         dtype=tf.float32,
    #         shape=[kernelSize, kernelSize, inputChannel, outputChannel],
    #         stddev=stddev),
    #     name=opName+'.kernel')
    # biasVariables = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[outputChannel]), name=opName+'.bias')
    # trainedVariables[opName + '.kernel'] = kernelVariables
    # trainedVariables[opName + '.bias'] = biasVariables
    # convolved = tf.nn.conv2d(inputLayer, kernelVariables, [1, stride, stride, 1], 'SAME', name=opName+'.convolved')
    # biased = tf.nn.bias_add(convolved, biasVariables, name=opName+'.biased')
    # return biased


def convolveReLU(opName, inputLayer, inputChannel, outputChannel, kernelSize, stride, stddev=1e-2):
    return ReLU(convolve(opName, inputLayer,
                         inputChannel, outputChannel,
                         kernelSize, stride, stddev),
                opName+'_rectified')


def convolveLeakyReLU(opName, inputLayer, inputChannel, outputChannel, kernelSize, stride, alpha=0.1, stddev=1e-2):
    return LeakyReLU(convolve(opName, inputLayer,
                              inputChannel, outputChannel,
                              kernelSize, stride, stddev),
                     alpha, opName+'_leakilyrectified')


def upconvolve(opName, inputLayer, inputChannel, outputChannel, kernelSize, stride, targetH, targetW, stddev=1e-2):
    return tflearn.layers.conv.conv_2d_transpose(inputLayer, outputChannel, kernelSize, [targetH, targetW], strides=stride,
                                                 padding='same', activation='linear', bias=False, scope=opName)
    # kernelVariables = tf.Variable(
    #     tf.truncated_normal(
    #         dtype=tf.float32,
    #         shape=[kernelSize, kernelSize, outputChannel, inputChannel],
    #         stddev=stddev),
    #     name=opName+'.kernel')
    # # bias is not used in upconvolutions in FlowNet.
    # # biasVariables = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[outputChannel]), name=opName+'.bias')
    # trainedVariables[opName + '.kernel'] = kernelVariables
    # upconvolved = tf.nn.conv2d_transpose(inputLayer,
    #     kernelVariables,
    #     tf.stack([tf.shape(inputLayer)[0], targetH, targetW, outputChannel]),
    #     [1, stride, stride, 1],
    #     'SAME', name=opName+'.upconvolved')
    # return upconvolved


def upconvolveReLU(opName, inputLayer, inputChannel, outputChannel, kernelSize, stride, targetH, targetW, stddev=1e-2):
    return ReLU(upconvolve(opName, inputLayer,
                           inputChannel, outputChannel,
                           kernelSize, stride,
                           targetH, targetW, stddev),
                opName+'_rectified')


def upconvolveLeakyReLU(opName, inputLayer, inputChannel, outputChannel, kernelSize, stride, targetH, targetW, alpha=0.1, stddev=1e-2):
    return LeakyReLU(upconvolve(opName, inputLayer,
                                inputChannel, outputChannel,
                                kernelSize, stride,
                                targetH, targetW, stddev),
                     alpha, opName+'_rectified')


def set_tf_keys(feed_dict, **kwargs):
    ret = dict([(k + ':0', v) for k, v in feed_dict.items()])
    ret.update([(k + ':0', v) for k, v in kwargs.items()])
    return ret


class Network:
    def __init__(self, name, trainable=True, reuse=None):
        self._built = reuse
        self._name = name
        self.trainable = trainable

    @property
    def name(self):
        return self._name

    def __call__(self, *args, **kwargs):
        with tf.variable_scope(self.name, reuse=self._built) as self.scope:
            self._built = True
            return self.build(*args, **kwargs)

    @property
    def trainable_variables(self):
        if isinstance(self.trainable, str):
            var_list = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name)
            return [var for var in var_list if re.fullmatch(self.trainable, var.name)]
        elif self.trainable:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name)
        else:
            return []

    @property
    def data_args(self):
        return dict()


class ParallelLayer:
    inputs = {}
    replicated_inputs = None


class MultiGPUs:
    def __init__(self, num):
        self.num = num

    def __call__(self, net, args, opt=None, scheme=None):
        args = [self.reshape(arg) for arg in args]
        results = []
        grads = []
        self.current_device = None
        for i in range(self.num):
            def auto_gpu(opr):
                # if opr.name.find('stack') != -1:
                #     print(opr)
                if opr.type.startswith('Gather') or opr.type in ('L2Loss', 'Pack', 'Gather', 'Tile', 'ReconstructionWrtImageGradient', 'Softmax', 'FloorMod', 'MatMul'):
                    return '/cpu:0'
                else:
                    return '/gpu:%d' % i
            with tf.device(auto_gpu):
                self.current_device = i
                net.controller = self
                result = net(*[arg[i] for arg in args])
                results.append(result)
                if opt is not None:
                    var_segment =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gaffdfrm/seg_stem")
                    var_deform =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gaffdfrm/deform_stem_0")
                    #registration network optimization
                    if scheme == 'reg' or scheme == 'reg_supervise':
                        grads.append(opt.compute_gradients(
                            result['loss'],var_list = var_deform))#result['loss'],var_list = net.trainable_variables))
                    #segmentation network optimization
                    else:
                        grads.append(opt.compute_gradients(
                            result['seg_loss'], var_list=var_segment))#previous    result['loss'],var_list = net.trainable_variables

        with tf.device('/gpu:0'):
            concat_result = {}
            for k in results[0]:
                if len(results[0][k].shape) == 0:
                    concat_result[k] = tf.stack(
                        [result[k] for result in results])
                else:
                    concat_result[k] = tf.concat(
                        [result[k] for result in results], axis=0)

            if grads:
                op = opt.apply_gradients(self.average_gradients(grads))
                return concat_result, op
            else:
                return concat_result

    def call(self, net, kwargs):
        if net.replicated_inputs is None:
            with tf.device('/gpu:0'):
                net.replicated_inputs = dict(
                    [(k, self.reshape(v)) for k, v in net.inputs.items()])
        for k, v in net.replicated_inputs.items():
            kwargs[k] = v[self.current_device]
        return net(**kwargs)

    @staticmethod
    def average_gradients(grads):
        ret = []
        for grad_list in zip(*grads):
            grad, var = grad_list[0]
            if grad is None:
                ret.append((None, var))
            else:
                print(var, var.device)
                ret.append(
                    (tf.add_n([grad for grad, _ in grad_list]) / len(grad_list), var))
        return ret

    def reshape(self, tensor):
        return tf.reshape(tensor, tf.concat([tf.stack([self.num, -1]), tf.shape(tensor)[1:]], axis = 0))


class FileRestorer:
    def __init__(self, rules=[(r'(.*)', r'\1')]):
        self.rules = rules

    def get_targets(self, key):
        targets = []
        for r in self.rules:
            if re.match(r[0], key):
                targets.append(re.sub(r[0], r[1], key))
        return targets

    def restore(self, sess, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            assign_ops = []
            g = sess.graph
            for key in sorted(var_to_shape_map):
                for target in self.get_targets(key):
                    var = None
                    try:
                        var = g.get_tensor_by_name(target + ':0')
                        print("restoring: {} ---> {}".format(key, target))
                    except KeyError as e:
                        print("Ignoring: {} ---> {}".format(key, target))
                    if var is not None:
                        assign_ops.append(
                            tf.assign(var, reader.get_tensor(key)))
            sess.run(assign_ops)
        except Exception as e:  # pylint: disable=broad-except
            raise(e)
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")
            if ("Data loss" in str(e) and
                    (any([e in file_name for e in [".index", ".meta", ".data"]]))):
                proposed_file = ".".join(file_name.split(".")[0:-1])
                v2_file_error_template = """
        It's likely that this is a V2 checkpoint and you need to provide the filename
        *prefix*.  Try removing the '.' and extension.  Try:
        inspect checkpoint --file_name = {}"""
                print(v2_file_error_template.format(proposed_file))


def restore_exists(sess, file_name, show=False):
    """Prints tensors in a checkpoint file.
    If no `tensor_name` is provided, prints the tensor names and shapes
    in the checkpoint file.
    If `tensor_name` is provided, prints the content of the tensor.
    Args:
      file_name: Name of the checkpoint file.
      tensor_name: Name of the tensor in the checkpoint file to print.
      all_tensors: Boolean indicating whether to print all tensors.
      all_tensor_names: Boolean indicating whether to print all tensor names.
    """
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        assign_ops = []
        if show:
            for key in sorted(var_to_shape_map):
                w = reader.get_tensor(key)
                print(key, w.dtype, w.shape)
        else:
            g = sess.graph
            for key in sorted(var_to_shape_map):
                try:
                    var = g.get_tensor_by_name(key + ':0')
                    print("restoring: ", key)
                except KeyError as e:
                    print("Ignoring: " + key)
                if var is not None:
                    assign_ops.append(tf.assign(var, reader.get_tensor(key)))
            sess.run(assign_ops)
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")
        if ("Data loss" in str(e) and
                (any([e in file_name for e in [".index", ".meta", ".data"]]))):
            proposed_file = ".".join(file_name.split(".")[0:-1])
            v2_file_error_template = """
It's likely that this is a V2 checkpoint and you need to provide the filename
*prefix*.  Try removing the '.' and extension.  Try:
inspect checkpoint --file_name = {}"""
            print(v2_file_error_template.format(proposed_file))
