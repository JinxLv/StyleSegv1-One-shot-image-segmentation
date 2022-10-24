import tensorflow as tf
import tflearn
from tensorflow.keras.layers import Conv3D, UpSampling3D, Activation, BatchNormalization,Conv3DTranspose,Add,Concatenate,Dropout
from tflearn.initializations import normal
from .spatial_transformer import Dense3DSpatialTransformer
from .layers import VecInt
from .utils import Network, ReLU, LeakyReLU,Softmax,Sigmoid
from .IN import InstanceNormalization
import tensorflow.keras.backend as K
import keras
from tensorflow.contrib.layers import instance_norm, layer_norm
#from keras.layers.core import Lambda

'''def channel_attention(input_feature, ratio=8, name=''):
    
    channel = input_feature.shape.as_list()[-1]
    Dense_1 = tf.layers.Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,bias_initializer='zeros',name = name+'_dense1')
    Dense_2 = tf.layers.Dense(channel,kernel_initializer='he_normal', use_bias=True,bias_initializer='zeros',name = name+'_dense2')
    Dense_3 = tf.layers.Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,bias_initializer='zeros',name = name+'_dense3')
    Dense_4 = tf.layers.Dense(channel, kernel_initializer='he_normal', use_bias=True,bias_initializer='zeros',name = name+'_dense4')
    avg_pool = tf.reduce_mean(input_feature,axis=(1,2,3))
    avg_pool = tf.reshape(avg_pool,(1,1,1,channel))
    print(avg_pool)
    avg_pool = Dense_1(avg_pool)
    print(avg_pool)
    avg_pool = Dense_2(avg_pool)
    max_pool = tf.reduce_max(input_feature,axis=(1,2,3))
    tf.reshape(max_pool,(1,1,1,channel))
    max_pool = Dense_3(max_pool)
    max_pool = Dense_4(max_pool)
    cbam_feature = avg_pool+max_pool
    cbam_feature = Activation('sigmoid')(cbam_feature)
    return cbam_feature
'''
def channel_attention(input_feature, name, ratio=8):    
    with tf.variable_scope(name):
        
        channel = input_feature.get_shape()[-1]
        avg_pool = tf.reduce_mean(input_feature, axis=[1,2,3], keepdims=True)

        avg_pool = tflearn.fully_connected(incoming = avg_pool,n_units = channel//ratio,activation = 'relu',name='mlp_0',scope = 'mlp_0',reuse=None)
        avg_pool = tflearn.fully_connected(incoming = avg_pool,n_units = channel,name='mlp_1',scope = 'mlp_1',reuse=None)

        max_pool = tf.reduce_max(input_feature, axis=[1,2,3], keepdims=True)
        max_pool = tflearn.fully_connected(incoming = max_pool,n_units = channel//ratio,activation = 'relu',name='mlp_0',scope = 'mlp_0',reuse=True)
        max_pool = tflearn.fully_connected(incoming = max_pool,n_units = channel,name='mlp_1',scope = 'mlp_1',reuse=True)

        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')
        
    return scale                           

def convolve(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False, weights_init='uniform_scaling'):
    return tflearn.layers.conv_3d(inputLayer, outputChannel, kernelSize, strides=stride,
                                  padding='same', activation='linear', bias=True, scope=opName, reuse=reuse, weights_init=weights_init)

def leakyReLU(inputLayer,opName, alpha = 0.1):
    return LeakyReLU(inputLayer,alpha,opName+'_leakilyrectified')

def inLeakyReLU(inputLayer,opName,alpha = 0.1):
    #IN = InstanceNormalization()(inputLayer)
    IN = instance_norm(inputLayer, scope=opName)## opName TO add  '_IN'
    return LeakyReLU(IN,alpha,opName+'_leakilyrectified')

def convInLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, alpha=0.1, stddev=1e-2, reuse=False):
    conv = convolve(opName, inputLayer,outputChannel, kernelSize, stride, stddev, reuse)
    #conv_In = InstanceNormalization()(conv)
    conv_In = instance_norm(conv, scope=opName+'_IN')
    return LeakyReLU(conv_In,alpha,opName+'_leakilyrectified')

def convolveReLU(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False):
    return ReLU(convolve(opName, inputLayer,
                         outputChannel,
                         kernelSize, stride, stddev=stddev, reuse=reuse),
                opName+'_rectified')

def convolveSoftmax(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False):
    return Softmax(convolve(opName, inputLayer,
                         outputChannel,
                         kernelSize, stride, stddev=stddev, reuse=reuse),
                opName+'_softmax')

def convolveSigmoid(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False):
    return Sigmoid(convolve(opName, inputLayer,
                         outputChannel,
                         kernelSize, stride, stddev=stddev, reuse=reuse),
                opName+'_sigmoid')

def convolveLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, alpha=0.1, stddev=1e-2, reuse=False):
    return LeakyReLU(convolve(opName, inputLayer,
                              outputChannel,
                              kernelSize, stride, stddev, reuse),
                     alpha, opName+'_leakilyrectified')

def upconvolve(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, stddev=1e-2, reuse=False, weights_init='uniform_scaling'):
    return tflearn.layers.conv.conv_3d_transpose(inputLayer, outputChannel, kernelSize, targetShape, strides=stride,
                                                 padding='same', activation='linear', bias=False, scope=opName, reuse=reuse, weights_init=weights_init)


""" def upconvolveInLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, stddev=1e-2, reuse=False):
    return LeakyReLU(InstanceNormalization()(upconvolve(opName, inputLayer,
                           outputChannel,
                           kernelSize, stride,
                           targetShape, stddev, reuse)),
                alpha,opName+'_rectified') """


def upconvolveLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, alpha=0.1, stddev=1e-2, reuse=False):
    return LeakyReLU(upconvolve(opName, inputLayer,
                                outputChannel,
                                kernelSize, stride,
                                targetShape, stddev, reuse),
                     alpha, opName+'_rectified')

def upconvolveInLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, alpha=0.1, stddev=1e-2, reuse=False):
    return LeakyReLU(upconvolve(opName, inputLayer,
                                outputChannel,
                                kernelSize, stride,
                                targetShape, stddev, reuse),
                     alpha, opName+'_rectified')

class FeatureNet(Network):
    def __init__(self, name, flow_multiplier=1., channels=16, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.use_vecint = False
        self.step = 3
        self.reconstruction = Dense3DSpatialTransformer()

        self.warping_featureMap = True
        self.use_warping_attention = True
        self.use_feature_attention = True
        
        #self.vecint = VecInt(method='ss',name = 'VTN_flow_int',int_steps= 2)

    def build(self, img1):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''

        #T1 Encoder
        dims = 3
        c = self.channels
        def resblock(inputLayer,opName,channel):
            residual = inputLayer
            conv1 = inLeakyReLU(inputLayer,opName)
            conv1_1_name  = opName[:opName.find('_')]+'_1'+opName[opName.find('_'):]
            print(conv1_1_name)
            conv1_1 = convolve(conv1_1_name,conv1, channel,   3, 1)
            add1 = Add()([conv1_1, residual])
            conv1_1 = inLeakyReLU(add1,conv1_1_name)
            return conv1_1
        

        conv0_fixed = convInLeakyReLU('conv0_fixed',   img1, c,   3, 1)# 160 * 160 * 160

        conv1_fixed = convolve('conv1_fixed',   conv0_fixed, 2*c,   3, 2)  # 80 * 80 * 80
        conv1_1_fixed = resblock(conv1_fixed,'conv1_fixed',2*c)
        
        conv2_fixed = convolve('conv2_fixed',   conv1_1_fixed,      4*c,   3, 2)  # 40 * 40 * 40
        conv2_1_fixed = resblock(conv2_fixed,'conv2_fixed',4*c)  # 40 * 40 * 40

        conv3_fixed = convolve('conv3_fixed',   conv2_1_fixed,      8*c,   3, 2)# 20 * 20 * 20
        conv3_1_fixed = resblock(conv3_fixed,'conv3_fixed',8*c)
        return conv0_fixed, conv1_1_fixed, conv2_1_fixed, conv3_1_fixed

class RWUNET(Network):
    def __init__(self, name, flow_multiplier=1., channels=8, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.use_vecint = False
        self.step = 3
        self.reconstruction = Dense3DSpatialTransformer()

        self.warping_featureMap = True
        self.use_warping_attention = True
        self.use_feature_attention = True
        
        #self.vecint = VecInt(method='ss',name = 'VTN_flow_int',int_steps= 2)

    def build(self, img1,img2):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''

        #T1 Encoder
        dims = 3
        c = self.channels
        def resblock(inputLayer,opName,channel):
            residual = inputLayer
            conv1 = inLeakyReLU(inputLayer,opName)
            conv1_1_name  = opName[:opName.find('_')]+'_1'+opName[opName.find('_'):]
            print(conv1_1_name)
            conv1_1 = convolve(conv1_1_name,conv1, channel,   3, 1)
            #add1 = Add()([conv1_1, residual])
            add1 = conv1_1+residual
            conv1_1 = inLeakyReLU(add1,conv1_1_name)
            return conv1_1
        

        conv0_fixed = convInLeakyReLU('conv0_fixed',   img1, c,   3, 1)# 160 * 160 * 160

        conv1_fixed = convolve('conv1_fixed',   conv0_fixed, 2*c,   3, 2)  # 80 * 80 * 80
        conv1_1_fixed = resblock(conv1_fixed,'conv1_fixed',2*c)
        
        conv2_fixed = convolve('conv2_fixed',   conv1_1_fixed,      4*c,   3, 2)  # 40 * 40 * 40
        conv2_1_fixed = resblock(conv2_fixed,'conv2_fixed',4*c)  # 40 * 40 * 40

        conv3_fixed = convolve('conv3_fixed',   conv2_1_fixed,      8*c,   3, 2)# 20 * 20 * 20
        conv3_1_fixed = resblock(conv3_fixed,'conv3_fixed',8*c)

        #T2 Encoder
        conv0_float = convInLeakyReLU('conv0_float',   img2, c,   3, 1)# 160 * 160 * 160

        conv1_float = convolve('conv1_float',   conv0_float, 2*c,   3, 2)  # 80 * 80 * 80
        conv1_1_float = resblock(conv1_float,'conv1_float',  2*c)  # 80 * 80 * 80

        conv2_float = convolve('conv2_float',   conv1_1_float,      4*c,   3, 2)  # 40 * 40 * 40
        conv2_1_float = resblock(conv2_float,'conv2_float',  4*c )  # 40 * 40 * 40

        conv3_float = convolve('conv3_float',   conv2_1_float,      8*c,   3, 2)# 20 * 20 * 20
        conv3_1_float = resblock(conv3_float,'conv3_float',   8*c)

        shape0 = conv0_float.shape.as_list()#160 [1,160,160,160,8]#
        shape1 = conv1_float.shape.as_list()#80 [1,80,80,80,16]#
        shape2 = conv2_float.shape.as_list()#40 [1,40,40,40,32]#
        shape3 = conv3_float.shape.as_list()#20 [1,20,20,20,64]#
        #shape4 = conv4_fixed.shape.as_list()#10 [1,10,10,10,128]#


        concat_bottleNeck = tf.concat([conv3_1_fixed,conv3_1_float],4,'concat_bottleNeck')
        concat_bottleNeck = convInLeakyReLU('conv_bottleNeck_contact_1', concat_bottleNeck,   8*c,  3, 1)
        concat_bottleNeck = convInLeakyReLU('conv_bottleNeck_contact_2', concat_bottleNeck,   8*c,  3, 1)

        predict_cache = []
        

        #   warping scale 2
        pred3 = convolve('pred3', concat_bottleNeck, dims, 3, 1)#10*10*10
        predict_cache.append(pred3)
        upsamp3to2 = upconvolve('upsamp3to2', pred3, dims, 4, 2, shape2[1:4])
        deconv2 = upconvolveInLeakyReLU('deconv2', concat_bottleNeck, shape2[4], 4, 2, shape2[1:4])
        
        if self.warping_featureMap:
            if self.use_warping_attention:
                warping_field_2 = self.AttentionModule_softmax(predict_cache)
            else:
                warping_field_2 = UpSampling3D()(pred3)
            conv2_1_float = self.reconstruction([conv2_1_float,warping_field_2])

        if self.use_feature_attention:
            fusion_fm_2 = self.FusionAttentionModule(conv2_1_fixed,conv2_1_float, deconv2,2)
            concat2 = tf.concat([fusion_fm_2, upsamp3to2],4, 'concat2') 
            #concat2 = self.FusionAttentionModule(conv2_1_fixed,conv2_1_float, deconv2,upsamp3to2,2)
        else:
            concat2 = tf.concat([conv2_1_fixed,conv2_1_float, deconv2, upsamp3to2],4, 'concat2') 

        #   warping scale 1
        #cost_2 = self.cost_volume(conv2_1_fixed,conv2_1_float,3,'cost_2')
        pred2 = convolve('pred2', concat2, dims, 3, 1)#20*20*20
        #pred2 = pred2 if not self.use_vecint else VecInt(method='ss',name = 'VTN_flow_int',int_steps= self.step)(pred2)
        predict_cache.append(pred2)
        upsamp2to1 = upconvolve('upsamp2to1', pred2, dims, 4, 2,shape1[1:4])
        deconv1 = upconvolveInLeakyReLU('deconv1', concat2, shape1[4], 4, 2, shape1[1:4])
        #deconv1 = UpSampling3D()(concat2)
        #deconv1 = convolveLeakyReLU('deconv1',   deconv1, shape1[4], 3, 1)

        if self.warping_featureMap:
            if self.use_warping_attention:
                warping_field_1 = self.AttentionModule_softmax(predict_cache)
            else:
                warping_field_1 = UpSampling3D()(pred2)
            conv1_1_float = self.reconstruction([conv1_1_float,warping_field_1])

        if self.use_feature_attention:
            fusion_fm_1 = self.FusionAttentionModule(conv1_1_fixed,conv1_1_float, deconv1,1)
            concat1 = tf.concat([fusion_fm_1, upsamp2to1], 4, 'concat1')
            #concat1 = self.FusionAttentionModule(conv1_1_fixed,conv1_1_float, deconv1,upsamp2to1,1)
        else:
            concat1 = tf.concat([conv1_1_fixed,conv1_1_float, deconv1, upsamp2to1], 4, 'concat1')
        
        #   warping scale 0
        #cost_1 = self.cost_volume(conv1_1_fixed,conv1_1_float,4,'cost_1')
        pred1 = convolve('pred1', concat1, dims, 3, 1)#80*80*80 
        #pred1 = pred1 if not self.use_vecint else VecInt(method='ss',name = 'VTN_flow_int',int_steps= self.step)(pred1)
        predict_cache.append(pred1)
        upsamp1to0 = upconvolve('upsamp1to0', pred1, dims, 4, 2, shape0[1:4])
        deconv0 = upconvolveInLeakyReLU('deconv0', concat1, shape0[4], 4, 2, shape0[1:4])

        if self.warping_featureMap:
            if self.use_warping_attention:
                warping_field_0 = self.AttentionModule_softmax(predict_cache)
            else:
                warping_field_0 = UpSampling3D()(pred1)
            conv0_float = self.reconstruction([conv0_float,warping_field_0])
            #concatFloatImgs = self.reconstruction([concatFloatImgs,warping_field_0])
        
        if self.use_feature_attention:
            fusion_fm_0 = self.FusionAttentionModule(conv0_fixed,conv0_float, deconv0,0)
            concat0 = tf.concat([fusion_fm_0, upsamp1to0], 4, 'concat0')
            #concat0 = self.FusionAttentionModule(conv0_fixed,conv0_float, deconv0,upsamp1to0,0)
        else:
            concat0 = tf.concat([conv0_fixed,conv0_float, deconv0, upsamp1to0], 4, 'concat0')

        concat0 = convolve('concat_conv', concat0, 8, 3, 1)
        pred0 = convolve('pred0', concat0, dims, 3, 1) #160*160*160
        pred0 = pred0 if not self.use_vecint else VecInt(method='ss',name = 'VTN_flow_int',int_steps= self.step)(pred0)
        #warping_field_final = pred0
        #progress_0 = self.reconstruction([warping_field_0,pred0])+pred0
        

        return {'flow': pred0}#,'flow_0':warping_field_0,'flow_1':warping_field_1,'flow_2':warping_field_2,'flow_3':warping_field_3}#,'flow_0':warping_field_0,'flow_1':warping_field_1,'flow_2':warping_field_2,'flow_3':warping_field_3}#,'flow_0':warping_field_0,'flow_1':warping_field_1,'flow_2':warping_field_2,'flow_3':warping_field_3}#,'flow_0':warping_field_0,'flow_1':warping_field_1,'flow_2':warping_field_2,'flow_3':warping_field_3,'flow_4':warping_field_4 }# ,'flow_1': warping_field_1,'flow_2':warping_field_2 ,'flow_3':warping_field_3

    def AttentionModule_softmax(self,prediction_list):
        list_num = len(prediction_list)
        level = 5 - list_num

        prediction_cache = []
        channel = 16
        for i,prediction in enumerate(prediction_list):
            prediction_cache.append(UpSampling3D(size = (2**(list_num-i),2**(list_num-i),2**(list_num-i)))(prediction))

        concatenate_0 = tf.concat(prediction_cache,4,'att_concat_{}_0'.format(level))
        conv_1 = convolve('att_conv_{}_1'.format(level),concatenate_0,channel,3,1)
        conv_2 = convolve('att_conv_{}_2'.format(level),conv_1,channel,5,1)
        
        weight_map_x = convolveSoftmax('att_conv_{}_x'.format(level),conv_2,list_num,3,1)
        weight_map_y = convolveSoftmax('att_conv_{}_y'.format(level),conv_2,list_num,3,1)
        weight_map_z = convolveSoftmax('att_conv_{}_z'.format(level),conv_2,list_num,3,1)

        for i,prediction in enumerate(prediction_cache):
            weight_map_cat = tf.concat([tf.expand_dims(weight_map_x[...,i],-1),tf.expand_dims(weight_map_y[...,i],-1),tf.expand_dims(weight_map_z[...,i],-1)],4,'att_concat_{}_cat'.format(level))
            prediction_cache[i] = tf.multiply(prediction,weight_map_cat)
            """ if i==0:
                progress_field = prediction_cache[-1] #self.reconstruction([warping_field_3,warping_field_2])+warping_field_2
            else:
                progress_field = self.reconstruction([progress_field,prediction_cache[-1]])+prediction_cache[-1] """
            #print(prediction_cache[i].shape.as_list(),'#'*30)
        #concatenate_1 = tf.concat(prediction_cache,4,'att_concat_{}_1'.format(level))
        add_result = tf.reduce_sum(prediction_cache,axis=0,name = 'add_{}_1'.format(level))
        return add_result

    def AttentionModule_sigmoid(self,prediction_list):
        list_num = len(prediction_list)
        level = 6 - list_num
        print(len(prediction_list))

        prediction_cache = []
        channel = 16
        for i,prediction in enumerate(prediction_list):
            #if i==5:
            #    prediction_cache.append(prediction)
            #else:
            prediction_cache.append(UpSampling3D(size = (2**(list_num-i),2**(list_num-i),2**(list_num-i)))(prediction))#*2**i
            
        concatenate_0 = tf.concat(prediction_cache,4,'att_concat_{}_0'.format(level))
        conv_1 = convolve('att_conv_{}_1'.format(level),concatenate_0,channel,3,1)
        conv_2 = convolve('att_conv_{}_2'.format(level),conv_1,channel,3,1)
        '''conv_1 = convolve('att_conv_{}_1'.format(level),concatenate_0,channel,1,1)
        conv_2 = convolve('att_conv_{}_2'.format(level),conv_1,channel,3,1)'''
        
        for i,prediction in enumerate(prediction_cache):
            weight_map = convolveSigmoid('att_weight_{}_{}'.format(level,i),conv_2,3,3,1)
            prediction_cache[i] = tf.multiply(prediction,weight_map)
            if i==0:
                progress_field = prediction_cache[-1] #self.reconstruction([warping_field_3,warping_field_2])+warping_field_2
            else:
                progress_field = self.reconstruction([progress_field,prediction_cache[-1]])+prediction_cache[-1]
        return progress_field

    def FusionAttentionModule(self,fixed_fm,float_fm,decon_fm,level):
        concatenate_fm = tf.concat([float_fm,fixed_fm,decon_fm],4,'att_fusion')
        channel_wise = channel_attention(concatenate_fm,name=str(level))#,'channel_wise_att_%d'%level)
        channel = fixed_fm.shape.as_list()[4]
        conv_1 = convolve('att_fusion_conv_{}_1'.format(level),concatenate_fm,channel,1,1)
        conv_2 = convolve('att_fusion_conv_{}_2'.format(level),conv_1,channel,3,1)
        weight_map = convolveSoftmax('att_fusion_conv__{}_3'.format(level),conv_2,3,3,1)
        
        concatenate_1 = tf.concat([tf.multiply(float_fm,tf.expand_dims(weight_map[...,0],-1)),tf.multiply(fixed_fm,tf.expand_dims(weight_map[...,1],-1)),tf.multiply(decon_fm,tf.expand_dims(weight_map[...,2],-1))],4,'att_fusion_concat_{}_1'.format(level))
        return concatenate_1*channel_wise

class RWUNET_v1(Network):
    def __init__(self, name, flow_multiplier=1., channels=16, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.use_vecint = False
        self.step = 7
        self.reconstruction = Dense3DSpatialTransformer()

        self.warping_featureMap = True
        self.use_warping_attention = True
        self.use_feature_attention = True
        
        #self.vecint = VecInt(method='ss',name = 'VTN_flow_int',int_steps= 2)

    def build(self, img1,img2):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''

        #T1 Encoder
        #img1 = AveragePooling3D()(img1)
        #img2 = AveragePooling3D()(img2)

        dims = 3
        c = self.channels
        def resblock(inputLayer,opName,channel):
            residual = inputLayer
            conv1 = inLeakyReLU(inputLayer,opName)
            conv1_1_name  = opName[:opName.find('_')]+'_1'+opName[opName.find('_'):]
            print(conv1_1_name)
            conv1_1 = convolve(conv1_1_name,conv1, channel,   3, 1)
            add1 = Add()([conv1_1, residual])
            conv1_1 = inLeakyReLU(add1,conv1_1_name)
            return conv1_1

        conv0_fixed = convInLeakyReLU('conv0_fixed',   img1, c,   3, 1)# 160 * 160 * 160

        conv1_fixed = convolve('conv1_fixed',   conv0_fixed, 2*c,   3, 2)  # 80 * 80 * 80
        conv1_1_fixed = resblock(conv1_fixed,'conv1_fixed',2*c)
        
        conv2_fixed = convolve('conv2_fixed',   conv1_1_fixed,      4*c,   3, 2)  # 40 * 40 * 40
        conv2_1_fixed = resblock(conv2_fixed,'conv2_fixed',4*c)  # 40 * 40 * 40

        conv3_fixed = convolve('conv3_fixed',   conv2_1_fixed,      8*c,   3, 2)# 20 * 20 * 20
        conv3_1_fixed = resblock(conv3_fixed,'conv3_fixed',8*c)

        #conv4_fixed = convolve('conv4_fixed',   conv3_1_fixed,    16*c,  3, 2)  # 10 * 10 * 10
        #conv4_1_fixed = resblock(conv4_fixed,'conv4_fixed',16*c)
        '''
        conv5_fixed = convolve('conv5_fixed',   conv4_1_fixed,    c*16,  3, 2)  # 5 * 5 * 5
        conv5_1_fixed = resblock(conv5_fixed, 'conv5_fixed', c*16)
        '''

        #T2 Encoder
        conv0_float = convInLeakyReLU('conv0_float',   img2, c,   3, 1)# 160 * 160 * 160

        conv1_float = convolve('conv1_float',   conv0_float, 2*c,   3, 2)  # 80 * 80 * 80
        conv1_1_float = resblock(conv1_float,'conv1_float',  2*c)  # 80 * 80 * 80

        conv2_float = convolve('conv2_float',   conv1_1_float,      4*c,   3, 2)  # 40 * 40 * 40
        conv2_1_float = resblock(conv2_float,'conv2_float',  4*c )  # 40 * 40 * 40

        conv3_float = convolve('conv3_float',   conv2_1_float,      8*c,   3, 2)# 20 * 20 * 20
        conv3_1_float = resblock(conv3_float,'conv3_float',   8*c)

        #conv4_float = convolve('conv4_float',   conv3_1_float,  16*c,  3, 2)  # 10 * 10 * 10
        #conv4_1_float = resblock(conv4_float,'conv4_float',   16*c)
        '''
        conv5_float = convolve('conv5_float',   conv4_1_float,    c*16,  3, 2)  # 5 * 5 * 5
        conv5_1_float = resblock(conv5_float,'conv5_float',   c*16)
        '''

        shape0 = conv0_float.shape.as_list()#160 [1,160,160,160,8]
        shape1 = conv1_float.shape.as_list()#80 [1,80,80,80,16]
        shape2 = conv2_float.shape.as_list()#40 [1,40,40,40,32]
        shape3 = conv3_float.shape.as_list()#20 [1,20,20,20,64]
        #shape4 = conv4_fixed.shape.as_list()#10 [1,10,10,10,128]


        concat_bottleNeck = tf.concat([conv3_1_fixed,conv3_1_float],4,'concat_bottleNeck')
        concat_bottleNeck = convInLeakyReLU('conv_bottleNeck_contact_1', concat_bottleNeck,   8*c,  3, 1)
        concat_bottleNeck = convInLeakyReLU('conv_bottleNeck_contact_2', concat_bottleNeck,   8*c,  3, 1)

        predict_cache = []
        
        #   warping scale 2
        
        pred3 = convolve('pred3', concat_bottleNeck, dims, 3, 1)#10*10*10
        pred3 = pred3 if not self.use_vecint else VecInt(method='ss',name = 'VTN_flow_int',int_steps= self.step)(pred3)
        predict_cache.append(pred3)
        #upsamp3to2 = upconvolve('upsamp3to2', pred3, dims, 4, 2, shape2[1:4])
        deconv2 = upconvolveInLeakyReLU('deconv2', concat_bottleNeck, shape2[4], 4, 2, shape2[1:4])
        #deconv2 = UpSampling3D()(concat3)
        #deconv2 = convolveLeakyReLU('deconv2',   deconv2, shape2[4], 3, 1)
        
        if self.warping_featureMap:
            if self.use_warping_attention:
                warping_field_2 = self.AttentionModule_sigmoid(predict_cache,'warp_2')
            else:
                warping_field_2 = UpSampling3D()(pred3)#(pred3)
            conv2_1_float = self.reconstruction([conv2_1_float,warping_field_2])

        if self.use_feature_attention:
            fusion_fm_2 = self.FusionAttentionModule(conv2_1_fixed,conv2_1_float, deconv2,2)
            concat2 = fusion_fm_2 #tf.concat([fusion_fm_2, upsamp3to2],4, 'concat2') 
            #concat2 = self.FusionAttentionModule(conv2_1_fixed,conv2_1_float, deconv2,upsamp3to2,2)
        else:
            concat2 = tf.concat([conv2_1_fixed,conv2_1_float, deconv2, upsamp3to2],4, 'concat2') 

        #   warping scale 1
        #cost_2 = self.cost_volume(conv2_1_fixed,conv2_1_float,3,'cost_2')
        pred2 = convolve('pred2', concat2, dims, 3, 1)#20*20*20
        pred2 = pred2 if not self.use_vecint else VecInt(method='ss',name = 'VTN_flow_int',int_steps= self.step)(pred2)
        predict_cache.append(pred2)
        #upsamp2to1 = upconvolve('upsamp2to1', pred2, dims, 4, 2,shape1[1:4])
        deconv1 = upconvolveInLeakyReLU('deconv1', concat2, shape1[4], 4, 2, shape1[1:4])
        #deconv1 = UpSampling3D()(concat2)
        #deconv1 = convolveLeakyReLU('deconv1',   deconv1, shape1[4], 3, 1)

        if self.warping_featureMap:
            if self.use_warping_attention:
                warping_field_1 = self.AttentionModule_sigmoid(predict_cache,'warp_1')
            else:
                warping_field_1 = UpSampling3D()(pred2)#UpSampling3D()(pred2)
            conv1_1_float = self.reconstruction([conv1_1_float,warping_field_1])

        if self.use_feature_attention:
            fusion_fm_1 = self.FusionAttentionModule(conv1_1_fixed,conv1_1_float, deconv1,1)
            concat1 = fusion_fm_1#tf.concat([fusion_fm_1, upsamp2to1], 4, 'concat1')
            #concat1 = self.FusionAttentionModule(conv1_1_fixed,conv1_1_float, deconv1,upsamp2to1,1)
        else:
            concat1 = tf.concat([conv1_1_fixed,conv1_1_float, deconv1, upsamp2to1], 4, 'concat1')
        
        #   warping scale 0
        #cost_1 = self.cost_volume(conv1_1_fixed,conv1_1_float,4,'cost_1')
        pred1 = convolve('pred1', concat1, dims, 3, 1)#80*80*80 
        pred1 = pred1 if not self.use_vecint else VecInt(method='ss',name = 'VTN_flow_int',int_steps= self.step)(pred1)
        predict_cache.append(pred1)
        #upsamp1to0 = upconvolve('upsamp1to0', pred1, dims, 4, 2, shape0[1:4])
        deconv0 = upconvolveInLeakyReLU('deconv0', concat1, shape0[4], 4, 2, shape0[1:4])

        if self.warping_featureMap:
            if self.use_warping_attention:
                warping_field_0 = self.AttentionModule_sigmoid(predict_cache,'warp_0')
            else:
                warping_field_0 = UpSampling3D()(pred1)#UpSampling3D()(pred1)
            conv0_float = self.reconstruction([conv0_float,warping_field_0])
            #concatFloatImgs = self.reconstruction([concatFloatImgs,warping_field_0])
        
        if self.use_feature_attention:
            fusion_fm_0 = self.FusionAttentionModule(conv0_fixed,conv0_float, deconv0,0)
            concat0 = fusion_fm_0#tf.concat([fusion_fm_0, upsamp1to0], 4, 'concat0')
            #concat0 = self.FusionAttentionModule(conv0_fixed,conv0_float, deconv0,upsamp1to0,0)
        else:
            concat0 = tf.concat([conv0_fixed,conv0_float, deconv0, upsamp1to0], 4, 'concat0')

        concat0 = convolve('concat_conv', concat0, 8, 3, 1)# 8-> 16
        pred0 = convolve('pred0', concat0, dims, 3, 1) #160*160*160
        pred0 = pred0 if not self.use_vecint else VecInt(method='ss',name = 'VTN_flow_int',int_steps= self.step)(pred0)
        #warping_field_final = pred0
        progress_0 = self.reconstruction([warping_field_0,pred0])+pred0
        print(progress_0.shape, img1.shape, img2.shape)

        return {'flow': progress_0}#,'flow_0':warping_field_0,'flow_1':warping_field_1,'flow_2':warping_field_2,'flow_3':warping_field_3}#,'flow_0':warping_field_0,'flow_1':warping_field_1,'flow_2':warping_field_2,'flow_3':warping_field_3}#,'flow_0':warping_field_0,'flow_1':warping_field_1,'flow_2':warping_field_2,'flow_3':warping_field_3}#,'flow_0':warping_field_0,'flow_1':warping_field_1,'flow_2':warping_field_2,'flow_3':warping_field_3,'flow_4':warping_field_4 }# ,'flow_1': warping_field_1,'flow_2':warping_field_2 ,'flow_3':warping_field_3

    def AttentionModule_softmax(self,prediction_list):
        list_num = len(prediction_list)
        level = 5 - list_num

        prediction_cache = []
        channel = 16
        for i,prediction in enumerate(prediction_list):
            prediction_cache.append(UpSampling3D(size = (2**(list_num-i),2**(list_num-i),2**(list_num-i)))(prediction))
            #prediction_cache.append(upsampling3d(prediction,scope='',scale=2**(list_num-i),interpolator='trilinear'))
                
        concatenate_0 = tf.concat(prediction_cache,4,'att_concat_{}_0'.format(level))
        conv_1 = convolve('att_conv_{}_1'.format(level),concatenate_0,channel,3,1)
        conv_2 = convolve('att_conv_{}_2'.format(level),conv_1,channel,5,1)
        
        weight_map_x = convolveSoftmax('att_conv_{}_x'.format(level),conv_2,list_num,3,1)
        weight_map_y = convolveSoftmax('att_conv_{}_y'.format(level),conv_2,list_num,3,1)
        weight_map_z = convolveSoftmax('att_conv_{}_z'.format(level),conv_2,list_num,3,1)

        for i,prediction in enumerate(prediction_cache):
            weight_map_cat = tf.concat([tf.expand_dims(weight_map_x[...,i],-1),tf.expand_dims(weight_map_y[...,i],-1),tf.expand_dims(weight_map_z[...,i],-1)],4,'att_concat_{}_cat'.format(level))
            prediction_cache[i] = tf.multiply(prediction,weight_map_cat)
            if i==0:
                progress_field = prediction_cache[-1] #self.reconstruction([warping_field_3,warping_field_2])+warping_field_2
            else:
                progress_field = self.reconstruction([progress_field,prediction_cache[-1]])+prediction_cache[-1]
            #print(prediction_cache[i].shape.as_list(),'#'*30)
        #concatenate_1 = tf.concat(prediction_cache,4,'att_concat_{}_1'.format(level))
        add_result = tf.reduce_sum(prediction_cache,axis=0,name = 'add_{}_1'.format(level))
        return progress_field#add_result

    
    def AttentionModule_sigmoid(self,prediction_list,scale):
        list_num = len(prediction_list)
        level = 5 - list_num
        print(len(prediction_list))

        prediction_cache = []
        channel = 16
        for i,prediction in enumerate(prediction_list):
            prediction_cache.append(UpSampling3D(size = (2**(list_num-i),2**(list_num-i),2**(list_num-i)))(prediction))#*2**i
            
        concatenate_0 = tf.concat(prediction_cache,4,'att_concat_{}_0'.format(level))
        conv_1 = convolve('att_conv_{}_1'.format(level),concatenate_0,channel*list_num,3,1)
        conv_2 = convolve('att_conv_{}_2'.format(level),conv_1,channel*list_num,3,1)
        
        for i,prediction in enumerate(prediction_cache):
            weight_map = convolveSigmoid('att_weight_{}_{}'.format(level,i),conv_2,3,3,1)
            prediction_cache[i] = tf.multiply(prediction,weight_map)
            if i==0:
                progress_field = prediction_cache[i] #self.reconstruction([warping_field_3,warping_field_2])+warping_field_2
            else:
                progress_field = self.reconstruction([progress_field,prediction_cache[i]])+prediction_cache[i]# -1 to i
            print(progress_field)
            
        return progress_field

    def FusionAttentionModule(self,fixed_fm,float_fm,decon_fm,level):
        concatenate_fm = tf.concat([float_fm,fixed_fm,decon_fm],4,'att_fusion')
        
        channel = fixed_fm.shape.as_list()[4]
        conv_1 = convolve('att_fusion_conv_{}_1'.format(level),concatenate_fm,channel,1,1)
        conv_2 = convolve('att_fusion_conv_{}_2'.format(level),conv_1,channel,3,1)
        weight_map = convolveSoftmax('att_fusion_conv__{}_3'.format(level),conv_2,3,3,1)
        
        concatenate_1 = tf.concat([tf.multiply(float_fm,tf.expand_dims(weight_map[...,0],-1)),tf.multiply(fixed_fm,tf.expand_dims(weight_map[...,1],-1)),\
            tf.multiply(decon_fm,tf.expand_dims(weight_map[...,2],-1))],4,'att_fusion_concat_{}_1'.format(level))
        channel_wise = channel_attention(concatenate_1,'channel_wise_att_%d'%level)#,'channel_wise_att_%d'%level)
        return concatenate_1*channel_wise
class DUAL(Network):
    def __init__(self, name, flow_multiplier=1., channels=8, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels
        self.reconstruction = Dense3DSpatialTransformer()

    def build(self, imgT1_fixed,imgT1_float,imgT2_fixed=None, imgT2_float=None):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        if imgT2_fixed==None:
            concatFixedImgs = imgT1_fixed
            concatFloatImgs = imgT1_float
        else:
            concatFixedImgs = tf.concat([imgT1_fixed, imgT2_fixed], 4, 'concatFixedImgs')
            concatFloatImgs = tf.concat([imgT1_float, imgT2_float], 4, 'concatFloatImgs')

        #T1 Encoder
        dims = 3
        c = self.channels
        def resblock(inputLayer,opName,channel,reuse = False):
            residual = inputLayer
            conv1_1 = convolve(opName+'_1',inputLayer, channel,   3, 1,reuse = reuse )
            conv1_2 = convolve(opName+'_2',conv1_1, channel,   3, 1,reuse = reuse )
            add1 = Add()([conv1_2, residual])
            return add1

        conv0_fixed = convolve('conv0_fixed',   concatFixedImgs, c,   3, 1)
        conv0_fixed = BatchNormalization()(conv0_fixed)
        conv0_fixed = Activation('relu')(conv0_fixed)

        conv1_fixed = convolve('conv1_fixed',   conv0_fixed, c,   3, 2)#80
        conv1_fixed = BatchNormalization()(conv1_fixed)
        conv1_fixed = Activation('relu')(conv1_fixed)
        conv1_fixed = convolve('conv1_fixed_',   conv1_fixed, c*2,   3, 1)  
        conv1_fixed = resblock(conv1_fixed,'conv1_fixed_1',c*2)
        conv1_fixed = resblock(conv1_fixed,'conv1_fixed_2',c*2)
        
        conv2_fixed = convolve('conv2_fixed',   conv1_fixed, c*2,   3, 2)#40
        conv2_fixed = BatchNormalization()(conv2_fixed)
        conv2_fixed = Activation('relu')(conv2_fixed)
        conv2_fixed = convolve('conv2_fixed_',   conv2_fixed, c*4,   3, 1) 
        conv2_fixed = resblock(conv2_fixed,'conv2_fixed_1',c*4)
        conv2_fixed = resblock(conv2_fixed,'conv2_fixed_2',c*4)
        
        conv3_fixed = convolve('conv3_fixed',   conv2_fixed, c*4,   3, 2)#20
        conv3_fixed = BatchNormalization()(conv3_fixed)
        conv3_fixed = Activation('relu')(conv3_fixed)
        conv3_fixed = convolve('conv3_fixed_',   conv3_fixed, c*4,   3, 1) 
        conv3_fixed = resblock(conv3_fixed,'conv3_fixed_1',c*4)
        conv3_fixed = resblock(conv3_fixed,'conv3_fixed_2',c*4)

        conv4_fixed = convolve('conv4_fixed',   conv3_fixed, c*4,   3, 2)#10

        #T2 Encoder
        conv0_float = convolve('conv0_float',   concatFloatImgs, c,   3, 1)
        conv0_float = BatchNormalization()(conv0_float)
        conv0_float = Activation('relu')(conv0_float)

        conv1_float = convolve('conv1_float',   conv0_float, c,   3, 2)#80
        conv1_float = BatchNormalization()(conv1_float)
        conv1_float = Activation('relu')(conv1_float)
        conv1_float = convolve('conv1_float_',   conv1_float, c*2,   3, 1)  
        conv1_float = resblock(conv1_float,'conv1_float_1',c*2)
        conv1_float = resblock(conv1_float,'conv1_float_2',c*2)
        
        conv2_float = convolve('conv2_float',   conv1_float, c*2,   3, 2)#40
        conv2_float = BatchNormalization()(conv2_float)
        conv2_float = Activation('relu')(conv2_float)
        conv2_float = convolve('conv2_float_',   conv2_float, c*4,   3, 1) 
        conv2_float = resblock(conv2_float,'conv2_float_1',c*4)
        conv2_float = resblock(conv2_float,'conv2_float_2',c*4)
        
        conv3_float = convolve('conv3_float',   conv2_float, c*4,   3, 2)#20
        conv3_float = BatchNormalization()(conv3_float)
        conv3_float = Activation('relu')(conv3_float)
        conv3_float = convolve('conv3_float_',   conv3_float, c*4,   3, 1) 
        conv3_float = resblock(conv3_float,'conv3_float_1',c*4)
        conv3_float = resblock(conv3_float,'conv3_float_2',c*4)

        conv4_float = convolve('conv4_float',   conv3_float, c*4,   3, 2)#10

        concat_bottleNeck = tf.concat([conv4_fixed,conv4_float],4,'concat_bottleNeck') 

        #   warping scale 3   
        pred4 = convolve('pred4', concat_bottleNeck, dims, 3, 1)
        warping_field_3 = UpSampling3D()(pred4)

        conv3_float_up = UpSampling3D()(conv4_float)
        conv3_fixed_up = UpSampling3D()(conv4_fixed)
        conv3_float_up = convolveLeakyReLU('decode3_conv1', conv3_float_up, c*4, 1, 1,reuse = None)
        conv3_fixed_up = convolveLeakyReLU('decode3_conv1', conv3_fixed_up, c*4, 1, 1,reuse = True)

        concat3_float = tf.concat([conv3_float,conv3_float_up], 4, 'concat3_float')
        concat3_fixed = tf.concat([conv3_fixed,conv3_fixed_up], 4, 'concat3_fixed')

        deconv3_float = convolveLeakyReLU('decode3', concat3_float, c*4, 3, 1,reuse = None)
        deconv3_fixed = convolveLeakyReLU('decode3', concat3_fixed, c*4, 3, 1,reuse = True)

        conv3_float_rc = self.reconstruction([deconv3_float,warping_field_3])
        concat_3_rc = tf.concat([conv3_float_rc,deconv3_fixed], 4, 'concat_3_rc')
        
        #   warping scale 2   
        pred3 = convolve('pred3', concat_3_rc, dims, 3, 1)
        warping_field_2 = UpSampling3D()(pred3)

        conv2_float_up = UpSampling3D()(conv3_float)
        conv2_fixed_up = UpSampling3D()(conv3_fixed)
        conv2_float_up = convolveLeakyReLU('decode2_conv1', conv2_float_up, c*4, 1, 1,reuse = None)
        conv2_fixed_up = convolveLeakyReLU('decode2_conv1', conv2_fixed_up, c*4, 1, 1,reuse = True)

        concat2_float = tf.concat([conv2_float,conv2_float_up], 4, 'concat2_float')
        concat2_fixed = tf.concat([conv2_fixed,conv2_fixed_up], 4, 'concat2_fixed')

        deconv2_float = convolveLeakyReLU('decode2', concat2_float, c*4, 3, 1,reuse = None)
        deconv2_fixed = convolveLeakyReLU('decode2', concat2_fixed, c*4, 3, 1,reuse = True)

        conv2_float_rc = self.reconstruction([deconv2_float,warping_field_2])
        concat_2_rc = tf.concat([conv2_float_rc,deconv2_fixed], 4, 'concat_2_rc')

        #   warping scale 1   
        pred2 = convolve('pred2', concat_2_rc, dims, 3, 1)
        warping_field_1 = UpSampling3D()(pred2)

        conv1_float_up = UpSampling3D()(conv2_float)
        conv1_fixed_up = UpSampling3D()(conv2_fixed)
        conv1_float_up = convolveLeakyReLU('decode1_conv1', conv1_float_up, c*2, 1, 1,reuse = None)
        conv1_fixed_up = convolveLeakyReLU('decode1_conv1', conv1_fixed_up, c*2, 1, 1,reuse = True)

        concat1_float = tf.concat([conv1_float,conv1_float_up], 4, 'concat1_float')
        concat1_fixed = tf.concat([conv1_fixed,conv1_fixed_up], 4, 'concat1_fixed')

        deconv1_float = convolveLeakyReLU('decode1', concat1_float, c*2, 3, 1,reuse = None)
        deconv1_fixed = convolveLeakyReLU('decode1', concat1_fixed, c*2, 3, 1,reuse = True)

        conv1_float_rc = self.reconstruction([deconv1_float,warping_field_1])
        concat_1_rc = tf.concat([conv1_float_rc,deconv1_fixed], 4, 'concat_1_rc')

        #   warping scale 0  
        pred1 = convolve('pred1', concat_1_rc, dims, 3, 1)
        warping_field_0 = UpSampling3D()(pred1)

        conv0_float_up = UpSampling3D()(conv1_float)
        conv0_fixed_up = UpSampling3D()(conv1_fixed)
        conv0_float_up = convolveLeakyReLU('decode0_conv1', conv0_float_up, c, 1, 1,reuse = None)
        conv0_fixed_up = convolveLeakyReLU('decode0_conv1', conv0_fixed_up, c, 1, 1,reuse = True)

        concat0_float = tf.concat([conv0_float,conv0_float_up], 4, 'concat0_float')
        concat0_fixed = tf.concat([conv0_fixed,conv0_fixed_up], 4, 'concat0_fixed')

        deconv0_float = convolveLeakyReLU('decode0', concat0_float, c, 3, 1,reuse = None)
        deconv0_fixed = convolveLeakyReLU('decode0', concat0_fixed, c, 3, 1,reuse = True)

        conv0_float_rc = self.reconstruction([deconv0_float,warping_field_0])
        concat_0_rc = tf.concat([conv0_float_rc,deconv0_fixed], 4, 'concat_0_rc')

        pred0 = convolve('pred0', concat_0_rc, dims, 3, 1)

        progress_3 = self.reconstruction([UpSampling3D()(pred4),pred3])+pred3
        progress_2 = self.reconstruction([UpSampling3D()(progress_3),pred2])+pred2
        progress_1 = self.reconstruction([UpSampling3D()(progress_2),pred1])+pred1
        progress_0 = self.reconstruction([UpSampling3D()(progress_1),pred0])+pred0
        #warping_field_final = convolve('final_flow', concat0, dims, 3, 1)

        return {'flow': progress_0}

class SegNet(Network):
    def __init__(self, name, flow_multiplier=1., channels=8, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels

    def build(self, imgT1_fixed,imgT1_float,imgT2_fixed, imgT2_float,seg_float):
        concatImgs = tf.concat([imgT1_fixed,imgT1_float,imgT2_fixed, imgT2_float,seg_float], 4, 'concatImgs_seg')
        print(concatImgs.shape.as_list())
        dims = 3
        c = self.channels
        current_layer = concatImgs
        levels = list()
        depth = 4
        pool_size = (2,2,2)
        # add levels with max pooling
        for layer_depth in range(depth):
            layer1 = self.create_convolution_block(input_layer=current_layer, n_filters=c*(2**layer_depth))
            layer2 = self.create_convolution_block(input_layer=layer1, n_filters=c*(2**layer_depth))
            if layer_depth < depth - 1:
                current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
                levels.append([layer1, layer2, current_layer])
            else:
                current_layer = layer2
                levels.append([layer1, layer2])

        # add levels with up-convolution or up-sampling
        for layer_depth in range(depth-2, -1, -1):
            up_convolution = self.get_up_convolution(pool_size=pool_size, is_deconv = False,
                                            n_filters=current_layer._keras_shape[-1])(current_layer)
            concat = tf.concat([up_convolution, levels[layer_depth][1]], axis=4)
            current_layer = self.create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1],
                                                 input_layer=concat)
            current_layer = self.create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1],
                                                 input_layer=current_layer)

        final_convolution = Conv3D(16, (1, 1, 1))(current_layer)
        act = Activation('softmax')(final_convolution)
        return act

    def create_convolution_block(self,input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=True):
        layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
        if batch_normalization:
            layer = BatchNormalization(axis=4)(layer)
        elif instance_normalization:
            try:
                from .IN import InstanceNormalization
            except ImportError:
                raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
            layer = InstanceNormalization(axis=4)(layer)
        if activation is None:
            return Activation('relu')(layer)
        else:
            return activation()(layer)
    def get_up_convolution(self,n_filters, pool_size, kernel_size=(2,2,2), strides=(2,2,2), is_deconv = False):
        if is_deconv:
            return Conv3DTranspose(filters=n_filters, kernel_size=kernel_size, strides=strides)
        else:
            return UpSampling3D(size = pool_size)

class SegNet1(Network):
    def __init__(self, name, flow_multiplier=1., channels=8, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels

    def build(self, imgT1_fixed,imgT1_float,imgT2_fixed, imgT2_float,seg_float):
        concatImgs = tf.concat([imgT1_fixed,imgT1_float,imgT2_fixed, imgT2_float,seg_float], 4, 'concatImgs_seg')
        print(concatImgs.shape.as_list())
        down_layer = []
        c = self.channels
        batch_normalization = True
        layer = self.res_block_v2_3d(concatImgs,c, batch_normalization=batch_normalization)
        down_layer.append(layer)
        layer = MaxPooling3D(pool_size=[2, 2, 2],  padding='same')(layer)
        print(str(layer.get_shape()))
        # down_layer_2
        layer = self.res_block_v2_3d(layer, 2*c, batch_normalization=batch_normalization)
        down_layer.append(layer)
        layer = MaxPooling3D(pool_size=[2, 2, 2], padding='same')(layer)
        print(str(layer.get_shape()))
        # down_layer_3
        layer = self.res_block_v2_3d(layer, 4*c, batch_normalization=batch_normalization)
        down_layer.append(layer)
        layer = MaxPooling3D(pool_size=[2, 2, 2],  padding='same')(layer)
        print(str(layer.get_shape()))
        # down_layer_4
        layer = self.res_block_v2_3d(layer, 8*c, batch_normalization=batch_normalization)
        down_layer.append(layer)
        layer = MaxPooling3D(pool_size=[2, 2, 2],  padding='same')(layer)
        print(str(layer.get_shape()))
        # bottle_layer
        layer = self.res_block_v2_3d(layer, 16*c, batch_normalization=batch_normalization)
        print(str(layer.get_shape()))
        # up_layer_4
        layer = self.up_and_concate_3d(layer, down_layer[3])
        layer = self.res_block_v2_3d(layer, 8*c, batch_normalization=batch_normalization)
        print(str(layer.get_shape()))
        # up_layer_3
        layer = self.up_and_concate_3d(layer, down_layer[2])
        layer = self.res_block_v2_3d(layer, 4*c, batch_normalization=batch_normalization)
        print(str(layer.get_shape()))
        # up_layer_2
        layer = self.up_and_concate_3d(layer, down_layer[1])
        layer = self.res_block_v2_3d(layer, 2*c, batch_normalization=batch_normalization)
        print(str(layer.get_shape()))
        # up_layer_1
        layer = self.up_and_concate_3d(layer, down_layer[0])
        layer = self.res_block_v2_3d(layer, c, batch_normalization=batch_normalization)
        print(str(layer.get_shape()))
        # score_layer
        layer = Conv3D(17, [1, 1, 1], strides=[1, 1, 1])(layer)
        print(str(layer.get_shape()))
        # softmax
        layer = Activation('softmax')(layer)
        print(str(layer.get_shape()))
        outputs = layer
        return outputs


    def res_block_v2_3d(self,input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3, 3], stride=[1, 1, 1],
                    padding='same'):

        input_n_filters = input_layer.get_shape().as_list()[4]
        #print(str(input_layer.get_shape()))
        #print(out_n_filters)
        #print(input_n_filters)
        layer = input_layer

        for i in range(2):
            if batch_normalization:
                layer = InstanceNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Conv3D(out_n_filters, kernel_size, strides=stride, padding=padding)(layer)

        if out_n_filters != input_n_filters:
            skip_layer = Conv3D(out_n_filters, [1, 1, 1], strides=stride, padding=padding)(input_layer)
        else:
            skip_layer = input_layer
        out_layer = add([layer, skip_layer])
        return out_layer

    def up_and_concate_3d(self,down_layer, layer):
        in_channel = down_layer.get_shape().as_list()[4]
        out_channel = in_channel // 2
        up = Conv3DTranspose(out_channel, [2, 2, 2], strides=[2, 2, 2], padding='valid')(down_layer)
        print("--------------")
        print(str(up.get_shape()))
        print(str(layer.get_shape()))
        print("--------------")
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=4))
        concate = my_concat([up, layer])
        return concate

class SegNet2(Network):
    def __init__(self, name, flow_multiplier=1., channels=8, seg_nums = 36, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.seg_nums = seg_nums

    def build(self, imgT1_fixed):
        #concatImgs = Concatenate()(
        #    concatImgs =  [imgT1_fixed, imgT1_float, imgT2_fixed, imgT2_float])
        #concatImgs = tf.concat([imgT1_fixed, imgT1_float, imgT2_fixed, imgT2_float], 4, 'concatImgs_seg')
        concatImgs = imgT1_fixed
        print(concatImgs.shape.as_list())
        down_layer = []
        c = self.channels
        slope = 1e-2
        rate = 0.6
        initial = 'he_normal'
        #encoder
        #scale 1
        layer = convolve('conv1_1', concatImgs, c, 3, 1)
        residual = layer
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = convolve('conv1_2', layer, c, 3, 1)
        layer = Dropout(rate=rate)(layer)
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = convolve('conv1_3', layer, c, 3, 1)
        layer = Add()([layer, residual])
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = instance_norm(layer, scope='ins_1')
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)
        down_layer.append(layer)

        #scale 2
        layer = convolve('conv2_1', layer, 2 * c, 3, 2)
        layer = self.res_block_v2_3d(layer, 2 * c, instance_normalization=True, name='res2')
        layer = instance_norm(layer, scope='ins2_1')
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)
        down_layer.append(layer)

        #scale 3
        layer = convolve('conv3_1', layer, 4 * c, 3, 2)
        layer = self.res_block_v2_3d(layer, 4 * c, instance_normalization=True, name='res3')
        layer = instance_norm(layer, scope='ins3_1')
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)
        down_layer.append(layer)

        #scale 4
        layer = convolve('conv4_1', layer, 8 * c, 3, 2)
        layer = self.res_block_v2_3d(layer, 8 * c, instance_normalization=True, name='res4')
        layer = instance_norm(layer, scope='ins4_1')
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)
        down_layer.append(layer)

        #scale 5
        layer = convolve('conv5_1', layer, 16 * c, 3, 2)
        layer = self.res_block_v2_3d(layer,
                                    16 * c,
                                    instance_normalization=True, name='res5')
        layer = instance_norm(layer, scope='ins5_1')
        layer_feature = tf.keras.layers.LeakyReLU(alpha=slope)(layer)

        #decoder
        #scale 4
        layer = UpSampling3D()(layer_feature)
        layer = convolve('conv4_d_1', layer, 8 * c, 3, 1)
        layer = instance_norm(layer, scope='ins4_d_1')
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = convolve('conv4_d_1_2', layer, 8 * c, 3, 1)
        layer = instance_norm(layer, scope='ins4_d_1_2')
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = Concatenate()([down_layer[-1], layer])
        layer = convolve('conv4_d_2', layer, 8 * c, 3, 1)
        layer = instance_norm(layer, scope='ins4_d_2')
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)
        output_4 = layer
        layer = convolve('conv4_d_2_2', layer, 8 * c, 3, 1)
        layer = instance_norm(layer, scope='ins4_d_2_2')
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)

        #scale 3
        layer = UpSampling3D()(layer)
        layer = convolve('conv3_d_1', layer, 4 * c, 3, 1)
        layer = instance_norm(layer, scope='ins3_d_1')
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = Concatenate()([down_layer[-2], layer])
        layer = convolve('conv3_d_2', layer, 4 * c, 3, 1)
        layer = instance_norm(layer, scope='ins3_d_2')
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)
        output_3 = layer
        layer = convolve('conv3_d_2_2', layer, 4 * c, 3, 1)
        layer = instance_norm(layer, scope='ins3_d_2_2')
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)

        #scale 2
        layer = UpSampling3D()(layer)
        layer = convolve('conv2_d_1', layer, 2 * c, 3, 1)
        layer = instance_norm(layer, scope='ins2_d_1')
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = Concatenate()([down_layer[-3], layer])
        layer = convolve('conv2_d_2', layer, 2 * c, 3, 1)
        layer = instance_norm(layer, scope='ins2_d_2')
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)
        output_2 = layer
        layer = convolve('conv2_d_2_2', layer, 2 * c, 3, 1)
        layer = instance_norm(layer, scope='ins2_d_2_2')
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)

        #scale 1
        layer = UpSampling3D()(layer)
        layer = convolve('conv1_d_1', layer, c, 3, 1)
        layer = instance_norm(layer, scope='ins1_d_1')
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = Concatenate()([down_layer[-4], layer])
        layer = convolve('conv1_d_2', layer, c, 3, 1)
        layer = instance_norm(layer, scope='ins1_d_2')
        layer = tf.keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = convolve('conv1_d_3', layer, self.seg_nums, 1, 1)

        #muti scale add 
        output_4 = convolve('multi_add_4', output_4, self.seg_nums, 1, 1)
        output_4to3 = UpSampling3D()(output_4)

        output_3 = convolve('multi_add_3', output_3, self.seg_nums, 1, 1)
        output_4add3 = output_3+output_4to3
        output_3to2 = UpSampling3D()(output_3)

        output_2 = convolve('multi_add_2', output_2, self.seg_nums, 1, 1)
        output_3add2 = output_2 + output_3to2
        output_2to1 = UpSampling3D()(output_3add2)
        
        final = layer+output_2to1

        prediction = Activation('softmax')(final)
        return prediction,final, down_layer[1]

    def attention_block(self, g, x,channel_num, name = 'attention', ):
        g1 = convolve(name+'_conv_g', g, channel_num, 1, 1)
        g1 = instance_norm(g1, scope=name+'_ins_g')
        x1 = convolve(name+'_conv_x', x, channel_num, 1, 1)
        x1 = instance_norm(x1, scope=name+'_ins_x')
        psi = tf.keras.layers.LeakyReLU(alpha=1e-2)(g1 + x1)
        psi = convolve(name+'_conv_psi', psi, 1, 1, 1)
        psi = instance_norm(psi, scope=name+'_ins_psi')
        psi = tf.sigmoid(psi)
        
        return x*psi
    
    def res_block_v2_3d(self,
                        input_layer,
                        out_n_filters,
                        instance_normalization=False,
                        kernel_size=[3, 3, 3],
                        stride=[1, 1, 1],
                        padding='same',
                        name = 'res'):

        input_n_filters = input_layer.get_shape().as_list()[4]
        layer = input_layer

        for i in range(2):
            if instance_normalization:
                layer = instance_norm(layer, scope=name+'_ins_1_'+str(i))
            layer = tf.keras.layers.LeakyReLU(alpha=1e-2)(layer)
            layer = convolve(name+'_conv_1_'+str(i), layer, out_n_filters, 3, 1)
            if i == 0:
                layer = Dropout(rate=0.6)(layer)
                #print('adding dropout!!!!')

        if out_n_filters != input_n_filters:
            skip_layer = convolve(name+'_conv_2', input_layer, out_n_filters, 1, 1)
        else:
            skip_layer = input_layer
        out_layer = Add()([layer, skip_layer])
        return out_layer

    def up_and_concate_3d(self, down_layer, layer, use_transpose=False):
        in_channel = down_layer.get_shape().as_list()[4]
        out_channel = in_channel // 2
        if use_transpose:
            up = Conv3DTranspose(out_channel, [2, 2, 2],
                                 strides=[2, 2, 2],
                                 padding='valid')(down_layer)
        else:
            up = UpSampling3D()(down_layer)
        print("--------------")
        print(str(up.get_shape()))
        print(str(layer.get_shape()))
        print("--------------")
        concate = Concatenate()([up, layer])
        return concate

class SegNet3(Network):
    def __init__(self, name, flow_multiplier=1., channels=8, seg_nums = 36, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.seg_nums = seg_nums

    def build(self, imgT1_fixed):
        #concatImgs = Concatenate()(
        #    concatImgs =  [imgT1_fixed, imgT1_float, imgT2_fixed, imgT2_float])
        #concatImgs = tf.concat([imgT1_fixed, imgT1_float, imgT2_fixed, imgT2_float], 4, 'concatImgs_seg')
        concatImgs = imgT1_fixed
        print(concatImgs.shape.as_list())
        down_layer = []
        c = self.channels
        slope = 1e-2
        rate = 0.6
        initial = 'he_normal'
        #encoder
        #scale 1
        layer = convolve('conv1_1', concatImgs, c, 3, 1)
        residual = layer
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = convolve('conv1_2', layer, c, 3, 1)
        layer = Dropout(rate=rate)(layer)
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = convolve('conv1_3', layer, c, 3, 1)
        layer = Add()([layer, residual])
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = instance_norm(layer, scope='ins_1')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        down_layer.append(layer)

        #scale 2
        layer = convolve('conv2_1', layer, 2 * c, 3, 2)
        layer = self.res_block_v2_3d(layer, 2 * c, instance_normalization=True, name='res2')
        layer = instance_norm(layer, scope='ins2_1')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        down_layer.append(layer)

        #scale 3
        layer = convolve('conv3_1', layer, 4 * c, 3, 2)
        layer = self.res_block_v2_3d(layer, 4 * c, instance_normalization=True, name='res3')
        layer = instance_norm(layer, scope='ins3_1')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        down_layer.append(layer)

        #scale 4
        layer = convolve('conv4_1', layer, 8 * c, 3, 2)
        layer = self.res_block_v2_3d(layer, 8 * c, instance_normalization=True, name='res4')
        layer = instance_norm(layer, scope='ins4_1')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        down_layer.append(layer)

        #scale 5
        layer = convolve('conv5_1', layer, 16 * c, 3, 2)
        layer = self.res_block_v2_3d(layer,
                                    16 * c,
                                    instance_normalization=True, name='res5')
        layer = instance_norm(layer, scope='ins5_1')
        layer_feature = keras.layers.LeakyReLU(alpha=slope)(layer)

        #decoder
        #scale 4
        layer = UpSampling3D()(layer_feature)
        layer = convolve('conv4_d_1', layer, 8 * c, 3, 1)
        layer = instance_norm(layer, scope='ins4_d_1')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = convolve('conv4_d_1_2', layer, 8 * c, 3, 1)
        layer = instance_norm(layer, scope='ins4_d_1_2')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        down_layer[-1] = self.attention_block(layer, down_layer[-1], 4 * c, 'atten_4')
        layer = Concatenate()([down_layer[-1], layer])
        layer = convolve('conv4_d_2', layer, 8 * c, 3, 1)
        layer = instance_norm(layer, scope='ins4_d_2')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        output_4 = layer
        layer = convolve('conv4_d_2_2', layer, 8 * c, 3, 1)
        layer = instance_norm(layer, scope='ins4_d_2_2')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)

        #scale 3
        layer = UpSampling3D()(layer)
        layer = convolve('conv3_d_1', layer, 4 * c, 3, 1)
        layer = instance_norm(layer, scope='ins3_d_1')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        down_layer[-2] = self.attention_block(layer, down_layer[-2], 2 * c, 'atten_3')
        layer = Concatenate()([down_layer[-2], layer])
        layer = convolve('conv3_d_2', layer, 4 * c, 3, 1)
        layer = instance_norm(layer, scope='ins3_d_2')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        output_3 = layer
        layer = convolve('conv3_d_2_2', layer, 4 * c, 3, 1)
        layer = instance_norm(layer, scope='ins3_d_2_2')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)

        #scale 2
        layer = UpSampling3D()(layer)
        layer = convolve('conv2_d_1', layer, 2 * c, 3, 1)
        layer = instance_norm(layer, scope='ins2_d_1')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        down_layer[-3] = self.attention_block(layer, down_layer[-3], 1 * c, 'atten_2')
        layer = Concatenate()([down_layer[-3], layer])
        layer = convolve('conv2_d_2', layer, 2 * c, 3, 1)
        layer = instance_norm(layer, scope='ins2_d_2')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        output_2 = layer
        layer = convolve('conv2_d_2_2', layer, 2 * c, 3, 1)
        layer = instance_norm(layer, scope='ins2_d_2_2')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)

        #scale 1
        layer = UpSampling3D()(layer)
        layer = convolve('conv1_d_1', layer, c, 3, 1)
        layer = instance_norm(layer, scope='ins1_d_1')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        down_layer[-4] = self.attention_block(layer, down_layer[-4], c, 'atten_1')
        layer = Concatenate()([down_layer[-4], layer])
        layer = convolve('conv1_d_2', layer, c, 3, 1)
        layer = instance_norm(layer, scope='ins1_d_2')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = convolve('conv1_d_3', layer, self.seg_nums, 1, 1)

        #muti scale add 
        output_4 = convolve('multi_add_4', output_4, self.seg_nums, 1, 1)
        output_4to3 = UpSampling3D()(output_4)

        output_3 = convolve('multi_add_3', output_3, self.seg_nums, 1, 1)
        output_4add3 = output_3+output_4to3
        output_3to2 = UpSampling3D()(output_3)

        output_2 = convolve('multi_add_2', output_2, self.seg_nums, 1, 1)
        output_3add2 = output_2 + output_3to2
        output_2to1 = UpSampling3D()(output_3add2)
        
        final = layer+output_2to1

        prediction = Activation('softmax')(final)
        return prediction,final, layer_feature

    def attention_block(self, g, x,channel_num, name = 'attention'):
        g1 = convolve(name+'_conv_g', g, channel_num, 1, 1)
        g1 = instance_norm(g1, scope=name+'_ins_g')
        x1 = convolve(name+'_conv_x', x, channel_num, 1, 1)
        x1 = instance_norm(x1, scope=name+'_ins_x')
        psi = keras.layers.LeakyReLU(alpha=1e-2)(g1 + x1)
        psi = convolve(name+'_conv_psi', psi, 1, 1, 1)
        psi = instance_norm(psi, scope=name+'_ins_psi')
        psi = tf.sigmoid(psi)
        
        return x*psi
    
    def res_block_v2_3d(self,
                        input_layer,
                        out_n_filters,
                        instance_normalization=False,
                        kernel_size=[3, 3, 3],
                        stride=[1, 1, 1],
                        padding='same',
                        name = 'res'):

        input_n_filters = input_layer.get_shape().as_list()[4]
        layer = input_layer

        for i in range(2):
            if instance_normalization:
                layer = instance_norm(layer, scope=name+'_ins_1_'+str(i))
            layer = keras.layers.LeakyReLU(alpha=1e-2)(layer)
            layer = convolve(name+'_conv_1_'+str(i), layer, out_n_filters, 3, 1)
            if i == 0:
                layer = Dropout(rate=0.6)(layer)
                #print('adding dropout!!!!')

        if out_n_filters != input_n_filters:
            skip_layer = convolve(name+'_conv_2', input_layer, out_n_filters, 1, 1)
        else:
            skip_layer = input_layer
        out_layer = Add()([layer, skip_layer])
        return out_layer

    def up_and_concate_3d(self, down_layer, layer, use_transpose=False):
        in_channel = down_layer.get_shape().as_list()[4]
        out_channel = in_channel // 2
        if use_transpose:
            up = Conv3DTranspose(out_channel, [2, 2, 2],
                                 strides=[2, 2, 2],
                                 padding='valid')(down_layer)
        else:
            up = UpSampling3D()(down_layer)
        print("--------------")
        print(str(up.get_shape()))
        print(str(layer.get_shape()))
        print("--------------")
        concate = Concatenate()([up, layer])
        return concate
class SegNet4(Network):
    def __init__(self, name, flow_multiplier=1., channels=8, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels

    def build(self, imgT1_fixed):
        #concatImgs = Concatenate()(
        #    concatImgs =  [imgT1_fixed, imgT1_float, imgT2_fixed, imgT2_float])
        #concatImgs = tf.concat([imgT1_fixed, imgT1_float, imgT2_fixed, imgT2_float], 4, 'concatImgs_seg')
        concatImgs = imgT1_fixed
        print(concatImgs.shape.as_list())
        down_layer = []
        c = self.channels
        slope = 1e-2
        rate = 0.6
        initial = 'he_normal'
        #encoder
        #scale 1
        layer = convolve('conv1_1', concatImgs, c, 3, 1)
        residual = layer
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = convolve('conv1_2', layer, c, 3, 1)
        layer = Dropout(rate=rate)(layer)
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = convolve('conv1_3', layer, c, 3, 1)
        layer = Add()([layer, residual])
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = instance_norm(layer, scope='ins_1')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        down_layer.append(layer)

        #scale 2
        layer = convolve('conv2_1', layer, 2 * c, 3, 2)
        layer = self.res_block_v2_3d(layer, 2 * c, instance_normalization=True, name='res2')
        layer = instance_norm(layer, scope='ins2_1')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        down_layer.append(layer)

        #scale 3
        layer = convolve('conv3_1', layer, 4 * c, 3, 2)
        layer = self.res_block_v2_3d(layer, 4 * c, instance_normalization=True, name='res3')
        layer = instance_norm(layer, scope='ins3_1')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        down_layer.append(layer)

        #scale 4
        layer = convolve('conv4_1', layer, 8 * c, 3, 2)
        layer = self.res_block_v2_3d(layer, 8 * c, instance_normalization=True, name='res4')
        layer = instance_norm(layer, scope='ins4_1')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        down_layer.append(layer)

        #scale 5
        layer = convolve('conv5_1', layer, 16 * c, 3, 2)
        layer = self.res_block_v2_3d(layer,
                                    16 * c,
                                    instance_normalization=True, name='res5')
        layer = instance_norm(layer, scope='ins5_1')
        layer_feature = keras.layers.LeakyReLU(alpha=slope)(layer)

        #decoder
        #scale 4
        layer = UpSampling3D()(layer_feature)
        layer = convolve('conv4_d_1', layer, 8 * c, 3, 1)
        layer = instance_norm(layer, scope='ins4_d_1')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = Concatenate()([down_layer[-1], layer])
        layer = convolve('conv4_d_2', layer, 8 * c, 3, 1)
        layer = instance_norm(layer, scope='ins4_d_2')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        output_4 = layer

        #scale 3
        layer = UpSampling3D()(layer)
        layer = convolve('conv3_d_1', layer, 4 * c, 3, 1)
        layer = instance_norm(layer, scope='ins3_d_1')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = Concatenate()([down_layer[-2], layer])
        layer = convolve('conv3_d_2', layer, 4 * c, 3, 1)
        layer = instance_norm(layer, scope='ins3_d_2')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        output_3 = layer

        #scale 2
        layer = UpSampling3D()(layer)
        layer = convolve('conv2_d_1', layer, 2 * c, 3, 1)
        layer = instance_norm(layer, scope='ins2_d_1')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = Concatenate()([down_layer[-3], layer])
        layer = convolve('conv2_d_2', layer, 2 * c, 3, 1)
        layer = instance_norm(layer, scope='ins2_d_2')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        output_2 = layer

        #scale 1
        layer = UpSampling3D()(layer)
        layer = convolve('conv1_d_1', layer, c, 3, 1)
        layer = instance_norm(layer, scope='ins1_d_1')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = Concatenate()([down_layer[-4], layer])
        layer = convolve('conv1_d_2', layer, c, 3, 1)
        layer = instance_norm(layer, scope='ins1_d_2')
        layer = keras.layers.LeakyReLU(alpha=slope)(layer)
        layer = convolve('conv1_d_3', layer, 36, 1, 1)

        #muti scale add 
        output_4 = convolve('multi_add_4', output_4, 36, 1, 1)
        output_4to3 = UpSampling3D()(output_4)

        output_3 = convolve('multi_add_3', output_3, 36, 1, 1)
        output_4add3 = output_3+output_4to3
        output_3to2 = UpSampling3D()(output_3)

        output_2 = convolve('multi_add_2', output_2, 36, 1, 1)
        output_3add2 = output_2 + output_3to2
        output_2to1 = UpSampling3D()(output_3add2)
        
        final = layer+output_2to1

        prediction = Activation('softmax')(final)
        return prediction,final, layer_feature

    def attention_block(self, g, x,channel_num, name = 'attention', ):
        g1 = convolve(name+'_conv_g', g, channel_num, 1, 1)
        g1 = instance_norm(g1, scope=name+'_ins_g')
        x1 = convolve(name+'_conv_x', x, channel_num, 1, 1)
        x1 = instance_norm(x1, scope=name+'_ins_x')
        psi = keras.layers.LeakyReLU(alpha=1e-2)(g1 + x1)
        psi = convolve(name+'_conv_psi', psi, 1, 1, 1)
        psi = instance_norm(psi, scope=name+'_ins_psi')
        psi = tf.sigmoid(psi)
        
        return x*psi
    
    def res_block_v2_3d(self,
                        input_layer,
                        out_n_filters,
                        instance_normalization=False,
                        kernel_size=[3, 3, 3],
                        stride=[1, 1, 1],
                        padding='same',
                        name = 'res'):

        input_n_filters = input_layer.get_shape().as_list()[4]
        layer = input_layer

        for i in range(2):
            if instance_normalization:
                layer = instance_norm(layer, scope=name+'_ins_1_'+str(i))
            layer = keras.layers.LeakyReLU(alpha=1e-2)(layer)
            layer = convolve(name+'_conv_1_'+str(i), layer, out_n_filters, 3, 1)
            if i == 0:
                layer = Dropout(rate=0.6)(layer)
                #print('adding dropout!!!!')

        if out_n_filters != input_n_filters:
            skip_layer = convolve(name+'_conv_2', input_layer, out_n_filters, 1, 1)
        else:
            skip_layer = input_layer
        out_layer = Add()([layer, skip_layer])
        return out_layer

    def up_and_concate_3d(self, down_layer, layer, use_transpose=False):
        in_channel = down_layer.get_shape().as_list()[4]
        out_channel = in_channel // 2
        if use_transpose:
            up = Conv3DTranspose(out_channel, [2, 2, 2],
                                 strides=[2, 2, 2],
                                 padding='valid')(down_layer)
        else:
            up = UpSampling3D()(down_layer)
        print("--------------")
        print(str(up.get_shape()))
        print(str(layer.get_shape()))
        print("--------------")
        concate = Concatenate()([up, layer])
        return concate
    
class VTN(Network):
    def __init__(self, name, flow_multiplier=1., channels=16, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels

    def build(self, imgT1_fixed,imgT1_float,imgT2_fixed, imgT2_float):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        concatImgs = tf.concat([imgT1_fixed,imgT1_float,imgT2_fixed, imgT2_float], 4, 'concatImgs')

        dims = 3
        c = self.channels
        conv1 = convolveLeakyReLU(
            'conv1',   concatImgs, c,   3, 2)  # 64 * 64 * 64
        conv2 = convolveLeakyReLU(
            'conv2',   conv1,      c*2,   3, 2)  # 32 * 32 * 32
        conv3 = convolveLeakyReLU('conv3',   conv2,      c*4,   3, 2)
        conv3_1 = convolveLeakyReLU('conv3_1', conv3,      c*4,   3, 1)
        conv4 = convolveLeakyReLU(
            'conv4',   conv3_1,    c*8,  3, 2)  # 16 * 16 * 16
        conv4_1 = convolveLeakyReLU('conv4_1', conv4,      c*8,  3, 1)
        conv5 = convolveLeakyReLU(
            'conv5',   conv4_1,    c*16,  3, 2)  # 8 * 8 * 8
        conv5_1 = convolveLeakyReLU('conv5_1', conv5,      c*16,  3, 1)
        conv6 = convolveLeakyReLU(
            'conv6',   conv5_1,    c*32,  3, 2)  # 4 * 4 * 4
        conv6_1 = convolveLeakyReLU('conv6_1', conv6,      c*32,  3, 1)
        # 16 * 32 = 512 channels

        shape0 = concatImgs.shape.as_list()
        shape1 = conv1.shape.as_list()
        shape2 = conv2.shape.as_list()
        shape3 = conv3.shape.as_list()
        shape4 = conv4.shape.as_list()
        shape5 = conv5.shape.as_list()
        shape6 = conv6.shape.as_list()

        pred6 = convolve('pred6', conv6_1, dims, 3, 1)
        upsamp6to5 = upconvolve('upsamp6to5', pred6, dims, 4, 2, shape5[1:4])
        deconv5 = upconvolveLeakyReLU(
            'deconv5', conv6_1, shape5[4], 4, 2, shape5[1:4])
        concat5 = tf.concat([conv5_1, deconv5, upsamp6to5], 4, 'concat5')

        pred5 = convolve('pred5', concat5, dims, 3, 1)
        upsamp5to4 = upconvolve('upsamp5to4', pred5, dims, 4, 2, shape4[1:4])
        deconv4 = upconvolveLeakyReLU(
            'deconv4', concat5, shape4[4], 4, 2, shape4[1:4])
        concat4 = tf.concat([conv4_1, deconv4, upsamp5to4],
                            4, 'concat4')  # channel = 512+256+2

        pred4 = convolve('pred4', concat4, dims, 3, 1)
        upsamp4to3 = upconvolve('upsamp4to3', pred4, dims, 4, 2, shape3[1:4])
        deconv3 = upconvolveLeakyReLU(
            'deconv3', concat4, shape3[4], 4, 2, shape3[1:4])
        concat3 = tf.concat([conv3_1, deconv3, upsamp4to3],
                            4, 'concat3')  # channel = 256+128+2

        pred3 = convolve('pred3', concat3, dims, 3, 1)
        upsamp3to2 = upconvolve('upsamp3to2', pred3, dims, 4, 2, shape2[1:4])
        deconv2 = upconvolveLeakyReLU(
            'deconv2', concat3, shape2[4], 4, 2, shape2[1:4])
        concat2 = tf.concat([conv2, deconv2, upsamp3to2],
                            4, 'concat2')  # channel = 128+64+2

        pred2 = convolve('pred2', concat2, dims, 3, 1)
        upsamp2to1 = upconvolve('upsamp2to1', pred2, dims, 4, 2, shape1[1:4])
        deconv1 = upconvolveLeakyReLU(
            'deconv1', concat2, shape1[4], 4, 2, shape1[1:4])
        concat1 = tf.concat([conv1, deconv1, upsamp2to1], 4, 'concat1')
        pred0 = upconvolve('upsamp1to0', concat1, dims, 4, 2, shape0[1:4])

        return {'flow': pred0 * 20 * self.flow_multiplier}

class VoxelMorph(Network):
    def __init__(self, name, flow_multiplier=1., channels=64, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.encoders = [m * channels for m in [1, 2, 2]]
        self.decoders = [m * channels for m in [2, 2, 2, 2,  1, 1]] + [3]

    def build(self, img1, img2):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        concatImgs = tf.concat([img1, img2], 4, 'concatImgs')

        conv1 = convolveLeakyReLU(
            'conv1',   concatImgs, self.encoders[0],     3, 2)  # 64 * 64 * 64
        conv2 = convolveLeakyReLU(
            'conv2',   conv1,      self.encoders[1],   3, 2)  # 32 * 32 * 32
        conv3 = convolveLeakyReLU(
            'conv3',   conv2,      self.encoders[2],   3, 2)  # 16 * 16 * 16

        net = convolveLeakyReLU('decode4', conv3, self.decoders[0], 3, 1)
        net = tf.concat([UpSampling3D()(net), conv2], axis=-1)
        net = convolveLeakyReLU('decode3',   net, self.decoders[1], 3, 1)
        net = tf.concat([UpSampling3D()(net), conv1], axis=-1)
        net = convolveLeakyReLU('decode2',   net, self.decoders[2], 3, 1)
        net = tf.concat([UpSampling3D()(net), concatImgs], axis=-1)
        net = convolveLeakyReLU('decode0',   net, self.decoders[-2], 3, 1)
        if len(self.decoders) == 8:
            net = convolveLeakyReLU('decode0_1', net, self.decoders[6], 3, 1)
        net = convolve(
            'flow', net, self.decoders[-1], 3, 1, weights_init=normal(stddev=1e-5))
        return {
            'flow': net * self.flow_multiplier
        }


def affine_flow(W, b, len1, len2, len3):
    b = tf.reshape(b, [-1, 1, 1, 1, 3])
    xr = tf.range(-(len1 - 1) / 2.0, len1 / 2.0, 1.0, tf.float32)
    xr = tf.reshape(xr, [1, -1, 1, 1, 1])
    yr = tf.range(-(len2 - 1) / 2.0, len2 / 2.0, 1.0, tf.float32)
    yr = tf.reshape(yr, [1, 1, -1, 1, 1])
    zr = tf.range(-(len3 - 1) / 2.0, len3 / 2.0, 1.0, tf.float32)
    zr = tf.reshape(zr, [1, 1, 1, -1, 1])
    wx = W[:, :, 0]
    wx = tf.reshape(wx, [-1, 1, 1, 1, 3])
    wy = W[:, :, 1]
    wy = tf.reshape(wy, [-1, 1, 1, 1, 3])
    wz = W[:, :, 2]
    wz = tf.reshape(wz, [-1, 1, 1, 1, 3])
    return (xr * wx + yr * wy) + (zr * wz + b)

def det3x3(M):
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    return tf.add_n([
                M[0][0] * M[1][1] * M[2][2],
                M[0][1] * M[1][2] * M[2][0],
                M[0][2] * M[1][0] * M[2][1]
            ]) - tf.add_n([
                M[0][0] * M[1][2] * M[2][1],
                M[0][1] * M[1][0] * M[2][2],
                M[0][2] * M[1][1] * M[2][0]
            ])

class VTNAffineStem(Network):
    def __init__(self, name, flow_multiplier=1.,channel = 8 ,**kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channel = channel

    def build(self, imgT1_fixed,imgT1_float,imgT2_fixed, imgT2_float):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        def resblock(inputLayer,opName,channel):
            residual = inputLayer
            conv1 = inLeakyReLU(inputLayer,opName)
            print(opName)
            conv1_1 = convolve(opName,conv1, channel,   3, 1)
            add1 = Add()([conv1_1, residual])
            conv1_1 = inLeakyReLU(add1,opName)
            return conv1_1

        concatImgs = tf.concat([imgT1_fixed,imgT1_float,imgT2_fixed, imgT2_float], 4, 'coloncatImgs_affine')

        dims = 3
        c = self.channel
        
        
        conv1 = convolveLeakyReLU(
            'conv1',   concatImgs, c,   3, 2)  # 64 * 64 * 64
        conv2 = convolveLeakyReLU(
            'conv2',   conv1,      2*c,   3, 2)  # 32 * 32 * 32
        conv3 = convolveLeakyReLU('conv3',   conv2,      4*c,   3, 2)
        conv3_1 = convolveLeakyReLU(
            'conv3_1', conv3,      4*c,   3, 1)
        conv4 = convolveLeakyReLU(
            'conv4',   conv3_1,    8*c,  3, 2)  # 16 * 16 * 16
        conv4_1 = convolveLeakyReLU(
            'conv4_1', conv4,      8*c,  3, 1)
        conv5 = convolveLeakyReLU(
            'conv5',   conv4_1,    16*c,  3, 2)  # 8 * 8 * 8
        conv5_1 = convolveLeakyReLU(
            'conv5_1', conv5,      16*c,  3, 1)
        conv6 = convolveLeakyReLU(
            'conv6',   conv5_1,    32*c,  3, 2)  # 4 * 4 * 4
        conv6_1 = convolveLeakyReLU(
            'conv6_1', conv6,      32*c,  3, 1)
        
        
        '''conv1 = convInLeakyReLU(
            'conv1',   concatImgs, c,   3, 2)  # 64 * 64 * 64
        conv2 = convInLeakyReLU(
            'conv2',   conv1,      2*c,   3, 2)  # 32 * 32 * 32
        conv3 = convolve('conv3',   conv2,      4*c,   3, 2)
        conv3_1 = resblock(
             conv3,  'conv3_1',    4*c)
        conv4 = convolve(
            'conv4',   conv3_1,    8*c,  3, 2)  # 16 * 16 * 16
        conv4_1 = resblock(
            conv4,  'conv4_1',     8*c)
        conv5 = convolve(
            'conv5',   conv4_1,    16*c,  3, 2)  # 8 * 8 * 8
        conv5_1 = resblock(
            conv5, 'conv5_1',      16*c)
        conv6 = convolve(
            'conv6',   conv5_1,    32*c,  3, 2)  # 4 * 4 * 4
        conv6_1 = resblock(
             conv6, 'conv6_1',     32*c)'''     

        ks = conv6_1.shape.as_list()[1:4]
        conv7_W = tflearn.layers.conv_3d(
            conv6_1, 9, ks, strides=1, padding='valid', activation='linear', bias=False, scope='conv7_W')
        conv7_b = tflearn.layers.conv_3d(
            conv6_1, 3, ks, strides=1, padding='valid', activation='linear', bias=False, scope='conv7_b')

        I = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        W = tf.reshape(conv7_W, [-1, 3, 3]) * self.flow_multiplier
        b = tf.reshape(conv7_b, [-1, 3]) * self.flow_multiplier
        A = W + I
        # the flow is displacement(x) = place(x) - x = (Ax + b) - x
        # the model learns W = A - I.

        sx, sy, sz = imgT1_float.shape.as_list()[1:4]
        flow = affine_flow(W, b, sx, sy, sz)
        # determinant should be close to 1
        det = det3x3(A)
        det_loss = tf.nn.l2_loss(det - 1.0)
        # should be close to being orthogonal
        # C=A'A, a positive semi-definite matrix
        # should be close to I. For this, we require C
        # has eigen values close to 1 by minimizing
        # k1+1/k1+k2+1/k2+k3+1/k3.
        # to prevent NaN, minimize
        # k1+eps + (1+eps)^2/(k1+eps) + ...
        eps = 1e-5
        epsI = [[[eps * elem for elem in row] for row in Mat] for Mat in I]
        C = tf.matmul(A, A, True) + epsI

        def elem_sym_polys_of_eigen_values(M):
            M = [[M[:, i, j] for j in range(3)] for i in range(3)]
            sigma1 = tf.add_n([M[0][0], M[1][1], M[2][2]])
            sigma2 = tf.add_n([
                M[0][0] * M[1][1],
                M[1][1] * M[2][2],
                M[2][2] * M[0][0]
            ]) - tf.add_n([
                M[0][1] * M[1][0],
                M[1][2] * M[2][1],
                M[2][0] * M[0][2]
            ])
            sigma3 = tf.add_n([
                M[0][0] * M[1][1] * M[2][2],
                M[0][1] * M[1][2] * M[2][0],
                M[0][2] * M[1][0] * M[2][1]
            ]) - tf.add_n([
                M[0][0] * M[1][2] * M[2][1],
                M[0][1] * M[1][0] * M[2][2],
                M[0][2] * M[1][1] * M[2][0]
            ])
            return sigma1, sigma2, sigma3
        s1, s2, s3 = elem_sym_polys_of_eigen_values(C)
        ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
        ortho_loss = tf.reduce_sum(ortho_loss)

        return {
            'flow': flow,
            'W': W,
            'b': b,
            'det_loss': det_loss,
            'ortho_loss': ortho_loss
        }
