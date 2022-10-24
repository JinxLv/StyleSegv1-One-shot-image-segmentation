import tensorflow as tf
import tflearn
import numpy as np
import keras.backend as K
from .utils import Network
from .base_networks import VTN, VoxelMorph, VTNAffineStem,RWUNET,SegNet2,DUAL,SegNet1,SegNet3,SegNet4,FeatureNet,RWUNET_v1
from .spatial_transformer import Dense3DSpatialTransformer, Fast3DTransformer
from .trilinear_sampler import TrilinearSampler
from .losses import NMI,Dice,NCC
from .layers import VecInt
from keras.layers.convolutional import AveragePooling3D
from keras.layers import  UpSampling3D
from .loss_functions import mixed_focal_loss, tversky_loss, focal_tversky_loss, combo_loss, cosine_tversky_loss, cross_entropy
from .data_augmentation import affine_intensity, random_affine, random_intensity, FDA_S2T, FDA_S2T2
from . import transform

def Grad(img):
    Filter_x = tf.convert_to_tensor([[[1,3,1],
                       [3,6,3],
                       [1,3,1]],
                      [[0,0,0],
                       [0,0,0],
                       [0,0,0]],
                       [[-1,-3,-1],
                       [-3,-6,-3],
                       [-1,-3,-1]]],dtype=tf.float32)
    Filter_y = tf.transpose(Filter_x,(1,0,2))
    Filter_z = tf.transpose(Filter_x,(2,1,0))

    Filter_x = tf.expand_dims(Filter_x,axis=-1)
    Filter_x= tf.expand_dims(Filter_x,axis=-1)
    Filter_y = tf.expand_dims(Filter_y,axis=-1)
    Filter_y= tf.expand_dims(Filter_y,axis=-1)
    Filter_z = tf.expand_dims(Filter_z,axis=-1)
    Filter_z= tf.expand_dims(Filter_z,axis=-1)

    output_x = tf.abs(tf.nn.conv3d(img, Filter_x, strides=[1, 1, 1, 1, 1], padding='SAME'))
    output_y = tf.abs(tf.nn.conv3d(img, Filter_y, strides=[1, 1, 1, 1, 1], padding='SAME'))
    output_z = tf.abs(tf.nn.conv3d(img, Filter_z, strides=[1, 1, 1, 1, 1], padding='SAME'))
    #output = output_x+output_y+output_z
    return output_x,output_y,output_z

def mask_metrics(seg1, seg2, img_size):
    ''' Given two segmentation seg1, seg2, 0 for background 255 for foreground.
    Calculate the Dice score 
    $ 2 * | seg1 \cap seg2 | / (|seg1| + |seg2|) $
    and the Jacc score
    $ | seg1 \cap seg2 | / (|seg1 \cup seg2|) $
    '''
    #sizes = np.prod(seg1.shape.as_list()[1:])
    seg1 = tf.reshape(seg1, [-1, img_size[0]*img_size[1]*img_size[2]])
    seg2 = tf.reshape(seg2, [-1, img_size[0]*img_size[1]*img_size[2]])
    seg1 = tf.cast(seg1 > 127.5, tf.float32)
    seg2 = tf.cast(seg2 > 127.5, tf.float32)
    #seg2 = tf.Print(seg2,[tf.reduce_sum(seg2, axis=-1)],'seg2_pos:')
    dice_score = 2.0 * tf.reduce_sum(seg1 * seg2, axis=-1) / (
        tf.reduce_sum(seg1, axis=-1) + tf.reduce_sum(seg2, axis=-1)+1e-8)
    
    union = tf.reduce_sum(tf.maximum(seg1, seg2), axis=-1)
    return (dice_score, tf.reduce_sum(tf.minimum(seg1, seg2), axis=-1) / tf.maximum(0.01, union))

class RecursiveCascadedNetworks(Network):
    default_params = {
        'weight': 1,
        'raw_weight': 1,
        'reg_weight': 1,
    }

    def __init__(self, name, framework,
                 base_network, n_cascades, rep=1,
                 det_factor=0.1, ortho_factor=0.1, reg_factor=1.0,
                 extra_losses={}, warp_gradient=True,
                 fast_reconstruction=False, warp_padding=False, scheme=None,
                 **kwargs):
        super().__init__(name)
        self.det_factor = det_factor
        self.ortho_factor = ortho_factor
        self.reg_factor = reg_factor
        self.scheme = scheme
        self.n_cascades = n_cascades

        self.base_network = eval(base_network)
        self.stems =  sum([
            [(self.base_network("deform_stem_" + str(i),
                                flow_multiplier=1.0 / n_cascades), {'raw_weight': 0})] * rep
            for i in range(n_cascades)], [])+ [(FeatureNet('feature',trainable = False),{'raw_weight': 0, 'reg_weight': 0})]+[(SegNet2('seg_stem',trainable = True, seg_nums = len(framework.segmentation_class_value.items())),{'raw_weight': 0, 'reg_weight': 0})]#TODO
        self.stems[n_cascades-1][1]['raw_weight'] = 1#TODO

        for _, param in self.stems:
            for k, v in self.default_params.items():
                if k not in param:
                    param[k] = v
        print(self.stems)
        
        self.framework = framework
        self.warp_gradient = warp_gradient
        self.fast_reconstruction = fast_reconstruction

        self.reconstruction = Fast3DTransformer(
            warp_padding) if fast_reconstruction else Dense3DSpatialTransformer(warp_padding)
        self.trilinear_sampler = TrilinearSampler()
        self.use_deepSuv = False
        self.output_mutiflow = False

    @property
    def trainable_variables(self):
        return list(set(sum([stem.trainable_variables for stem, params in self.stems], [])))

    @property
    def data_args(self):
        return dict()

    def build(self, imgT1_fixed,imgT1_float, seg1, seg2, pt1, pt2, pseudo_label):
        
        def CosineDisMap(I,J):
            amb = lambda x:tf.pow(tf.reduce_sum(x*x,axis=-1, keep_dims=True),0.5)
            distmap = 1 - tf.abs(tf.reduce_sum(I*J,axis=-1, keep_dims=True))/(amb(I)*amb(J))
            return distmap
        
        def GetSimilarityLoss(T1_fixed,T1_warped,scale,only_label=False,flow=None):
            NCC_loss = NCC(win=[9,9,9])
            ncc_loss = NCC_loss.loss(T1_fixed,T1_warped)
            loss_similarity = ncc_loss
            return loss_similarity,ncc_loss
        
        def mask_class(seg, value):
            return tf.cast(tf.abs(seg - value) < 0.5, tf.float32) * 255
        
        def one_hot_seg(s):
            ret = []
            for k, v in self.framework.segmentation_class_value.items():
                ret_class = mask_class(s, v)
                ret.append(ret_class[...,0])
            ret = tf.stack(ret, axis=-1)
            return ret
        
        stem_results = []
        stem_result = self.stems[0][0](imgT1_fixed,imgT1_float)

        stem_result['warpedT1'] = self.reconstruction([imgT1_float, stem_result['flow']])
        stem_result['agg_flow'] = stem_result['flow']
        stem_results.append(stem_result)

        for stem, params in self.stems[1:self.n_cascades]:
            stem_result = stem(imgT1_fixed, stem_results[-1]['warpedT1'])
            stem_result['agg_flow'] = self.reconstruction(
                    [stem_results[-1]['agg_flow'], stem_result['flow']]) + stem_result['flow']
            stem_result['warpedT1'] = self.reconstruction(
                [imgT1_float, stem_result['agg_flow']])
            stem_results.append(stem_result)
       
        for stem_result, (stem, params) in zip(stem_results, self.stems):
            #affine network
            if 'W' in stem_result:
                stem_result['loss'] = stem_result['det_loss'] * \
                    self.det_factor + \
                    stem_result['ortho_loss'] * self.ortho_factor
                if params['raw_weight'] > 0:
                    stem_result['raw_loss'],ncc_loss = GetSimilarityLoss(imgT1_fixed,stem_result['warpedT1'], 0)
                    stem_result['loss'] = stem_result['loss'] + \
                        stem_result['raw_loss'] * params['raw_weight']
            else:
                #non-rigid network
                if params['raw_weight'] > 0:
                    if not self.use_deepSuv:
                        stem_result['raw_loss'],ncc_loss = GetSimilarityLoss(imgT1_fixed,stem_result['warpedT1'],0,False,flow=stem_result['agg_flow'])

                if params['reg_weight'] > 0:
                    stem_result['reg_loss'] = self.regularize_loss(
                        stem_result['flow']) * self.reg_factor
                stem_result['loss'] = sum(
                    [stem_result[k] * params[k.replace('loss', 'weight')] for k in stem_result if k.endswith('loss')])

        ret = {}
        
        flow = stem_results[-1]['agg_flow']
        warped_T1 = stem_results[-1]['warpedT1']
        jacobian_det = self.jacobian_det(flow)
        loss = sum([r['loss'] * params['weight']
                    for r, (stem, params) in zip(stem_results, self.stems)])
        stem_results[-1]['ncc_loss'] = ncc_loss
        
        pt_mask1 = tf.reduce_any(tf.reduce_any(pt1 >= 0, -1), -1)
        pt_mask2 = tf.reduce_any(tf.reduce_any(pt2 >= 0, -1), -1)
        pt1 = tf.maximum(pt1, 0.0)
        moving_pt1 = pt1 + self.trilinear_sampler([flow, pt1])
        pt_mask = tf.cast(pt_mask1, tf.float32) * tf.cast(pt_mask2, tf.float32)
        landmark_dists = tf.sqrt(tf.reduce_sum(
            (moving_pt1 - pt2) ** 2, axis=-1)) * tf.expand_dims(pt_mask, axis=-1)
        landmark_dist = tf.reduce_mean(landmark_dists, axis=-1)
    
        if self.framework.segmentation_class_value is None:
            seg_fixed = seg1
            warped_seg_moving = self.reconstruction([seg2, flow])
            dice_score, jacc_score = mask_metrics(seg_fixed, warped_seg_moving, self.framework.image_size)
            jaccs = [jacc_score]
            dices = [dice_score]
        else:
            jaccs = []
            dices = []
            fixed_segs = []
            warped_segs = []
            for k, v in self.framework.segmentation_class_value.items():
                print('Segmentation {}, {}'.format(k, v))
                fixed_seg_class = mask_class(seg1, v)
                warped_seg_class = self.reconstruction(
                    [mask_class(seg2, v), flow])
                class_dice, class_jacc = mask_metrics(
                    fixed_seg_class, warped_seg_class, self.framework.image_size)
                ret['jacc_{}'.format(k)] = class_jacc
                jaccs.append(class_jacc)
                dices.append(class_dice)
                fixed_segs.append(fixed_seg_class[...,0])
                warped_segs.append(warped_seg_class[...,0])
            seg_fixed = tf.stack(fixed_segs, axis=-1)
            warped_seg_moving = tf.stack(warped_segs, axis=-1)
            dice_score1, jacc_score = tf.add_n(
                dices[1:]) / (len(dices)-1), tf.add_n(jaccs) / len(jaccs)
        # ret["dices2"] = tf.stack(dices[1:], axis=-1)
        
        f0, f1, f2, f3= self.stems[-2][0](imgT1_fixed)
        ret['feature'] = f0
        f0_, f1_, f2_, f3_ = self.stems[-2][0](warped_T1)
        ret['feature1'] = f0_
        #seg_loss = 10*tf.reduce_mean((feature-feature1)**2)
        ret['perceptual_loss0'] = NCC().loss(f0,f0_)
        ret['perceptual_loss1'] = NCC().loss(f1,f1_)
        ret['perceptual_loss2'] = NCC().loss(f2,f2_)
        ret['perceptual_loss3'] = NCC().loss(f3,f3_)
        #ret['distmap'] = ret['perceptual_loss0']+0.5*ret['perceptual_loss1']+0.5*ret['perceptual_loss2']
        ret['distmap'] = CosineDisMap(f0, f0_)
        ret['cosin_dist_loss'] = tf.reduce_mean(ret['distmap'])
        
        if self.scheme == "reg":
            loss += ret['cosin_dist_loss']
        if self.scheme == "reg_supervise":
            seg_result, _, _ = self.stems[-1][0](imgT1_fixed)
            seg_loss = Dice().loss(warped_seg_moving/255.0,seg_result)
            ret['seg_loss0'] = seg_loss
            loss = loss + ret['cosin_dist_loss'] + ret['seg_loss0']   
        if self.scheme == 'seg':
            pseudo_label =  one_hot_seg(pseudo_label)
            # pseudo_label = warped_seg_moving    
            warped_T1_s2t = FDA_S2T2(warped_T1, imgT1_fixed, if_random=True)
            seg_result1,_, _ = self.stems[-1][0](warped_T1_s2t)
            seg_result,_, _ = self.stems[-1][0](imgT1_fixed)
            ret['target_loss'] = combo_loss()(pseudo_label/255.0,seg_result)
            ret['atlas_loss'] = combo_loss()(warped_seg_moving/255.0,seg_result1)
            seg_loss = ret['atlas_loss'] + 0.1*ret['target_loss']
            dices2 = self.Get_Dices(seg_fixed/255.0,seg_result)
            dice_score2 = tf.add_n(dices2) / len(dices2)
            #固定图预测标签与伪标签的多类别dice_loss
            dices3 = self.Get_Dices(pseudo_label/255.0,seg_result)
            #固定图预测标签与固定图真实标签的多类别dice_loss
            dices_pseudo = self.Get_Dices(seg_fixed/255.0,pseudo_label/255.0)
            ret['dices_pseudo']  = tf.stack(dices_pseudo, axis=-1)
            ret['dices3']  = tf.stack(dices3, axis=-1)
            ret['warped_T1_s2t'] = warped_T1_s2t * 255.0
            ret["seg_loss"] = tf.reshape(seg_loss, (1, ))
            ret["dice_score2"] = dice_score2
            ret["dices2"] = tf.stack(dices2, axis=-1)
            ret['seg_result'] = seg_result*255

        ret.update({'loss': tf.reshape(loss, (1, )),
                    'dice_score1': dice_score1,
                    'jacc_score': jacc_score,
                    'dices': tf.stack(dices, axis=-1),
                    'jaccs': tf.stack(jaccs, axis=-1),
                    'landmark_dist': landmark_dist,
                    'landmark_dists': landmark_dists,
                    'real_flow': flow,
                    'seg1':seg1,
                    'pt_mask': pt_mask,
                    'reconstruction_T1': warped_T1 * 255.0,
                    'warped_moving_T1': warped_T1 * 255.0,
                    'seg_fixed': seg_fixed,
                    'warped_seg_moving': warped_seg_moving,
                    'image_fixed_T1': imgT1_fixed*255,
                    'image_float_T1':imgT1_float*255,
                    'moving_pt': moving_pt1,
                    'jacobian_det': jacobian_det,
                    'MSE_score':tf.reduce_mean(tf.abs(imgT1_fixed-warped_T1)*255.0),
                    'NMI_score':NMI(bin_centers =np.linspace(0,1,16), vol_size=tuple(self.framework.image_size)).global_mi(imgT1_fixed, warped_T1),
                    'ncc_score':self.similarity_loss(imgT1_fixed,warped_T1)})
        
        #every step returns
        for i, r in enumerate(stem_results):
            for k in r:
                if k.endswith('loss'):
                    ret['{}_{}'.format(i, k)] = r[k]
            ret['warped_seg_moving_%d' %
                i] = self.reconstruction([seg2, r['agg_flow']])
            ret['warped_moving_T1_%d' % i] = r['warpedT1']
            ret['flow_%d' % i] = r['flow']
            ret['real_flow_%d' % i] = r['agg_flow']
        return ret

    def similarity_loss(self, img1, warped_img2):
        #sizes = np.prod(img1.shape.as_list()[1:])
        flatten1 = tf.reshape(img1, [-1, self.framework.image_size[0]*self.framework.image_size[1]*self.framework.image_size[2]])
        flatten2 = tf.reshape(warped_img2, [-1, self.framework.image_size[0]*self.framework.image_size[1]*self.framework.image_size[2]])

        if self.fast_reconstruction:
            _, pearson_r, _ = tf.user_ops.linear_similarity(flatten1, flatten2)
        else:
            mean1 = tf.reshape(tf.reduce_mean(flatten1, axis=-1), [-1, 1])
            mean2 = tf.reshape(tf.reduce_mean(flatten2, axis=-1), [-1, 1])
            var1 = tf.reduce_mean(tf.square(flatten1 - mean1), axis=-1)
            var2 = tf.reduce_mean(tf.square(flatten2 - mean2), axis=-1)
            cov12 = tf.reduce_mean(
                (flatten1 - mean1) * (flatten2 - mean2), axis=-1)
            pearson_r = cov12 / tf.sqrt((var1 + 1e-6) * (var2 + 1e-6))

        #raw_loss = 1 - pearson_r
        raw_loss = tf.reduce_sum(pearson_r)
        return raw_loss


    def regularize_loss(self, flow):
        ret = ((tf.nn.l2_loss(flow[:, 1:, :, :] - flow[:, :-1, :, :]) +
                tf.nn.l2_loss(flow[:, :, 1:, :] - flow[:, :, :-1, :]) +
                tf.nn.l2_loss(flow[:, :, :, 1:] - flow[:, :, :, :-1])) / np.prod([128,128,128,3]))#flow.shape.as_list()[1:5]
        return ret

    def jacobian_det(self, flow):
        _, var = tf.nn.moments(tf.linalg.det(tf.stack([
            flow[:, 1:, :-1, :-1] - flow[:, :-1, :-1, :-1] +
            tf.constant([1, 0, 0], dtype=tf.float32),
            flow[:, :-1, 1:, :-1] - flow[:, :-1, :-1, :-1] +
            tf.constant([0, 1, 0], dtype=tf.float32),
            flow[:, :-1, :-1, 1:] - flow[:, :-1, :-1, :-1] +
            tf.constant([0, 0, 1], dtype=tf.float32)
        ], axis=-1)), axes=[1, 2, 3])
        return tf.sqrt(var)

    def Get_Dices(self, seg1,seg2):
        dices = []
        i = 0
        for k, v in self.framework.segmentation_class_value.items() :
            #class_dice = - Dice().loss( tf.expand_dims(seg1[...,i], -1), tf.expand_dims(seg2[...,i], -1) )
            if i == 0:
                i += 1
                continue
            class_dice, class_jacc = mask_metrics(
                    seg1[..., i]*255, seg2[..., i]*255, self.framework.image_size)
            dices.append(class_dice)
            i += 1
        return dices
    
    def Get_box(self,seg,padding = 5):
        shape = [1]+self.framework.image_size[1]
        index = tf.cast(tf.where(seg>0),tf.int32)
        print('index shape:', index.get_shape().as_list())
        x_min = tf.reduce_min(index[:,1])
        x_max = tf.reduce_max(index[:,1])
        y_min = tf.reduce_min(index[:,2])
        y_max = tf.reduce_max(index[:,2])
        z_min = tf.reduce_min(index[:,3])
        z_max = tf.reduce_max(index[:,3])
        print([x_min,x_max,y_min,y_max,z_min,z_max])
        return [x_min,x_max,y_min,y_max,z_min,z_max]

    def crop_resize(self, img, box, w=64, h=64, d=96):
        center = [(box[0]+box[1])//2,
                  (box[2]+box[3])//2,
                  (box[4]+box[5])//2,]
        croped = img[:,
                    center[0]-w//2:center[0]+w//2,
                    center[1]-h//2:center[1]+h//2,
                    center[2]-d//2:center[2]+d//2,
                    :]
        return croped
    
    
