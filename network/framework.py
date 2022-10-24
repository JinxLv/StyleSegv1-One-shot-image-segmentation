import tensorflow as tf
import numpy as np
import tflearn
from tqdm import tqdm
from . import transform
from .utils import MultiGPUs
from .spatial_transformer import Dense3DSpatialTransformer, Fast3DTransformer
from .recursive_cascaded_networks import RecursiveCascadedNetworks
from .AdaBelief_tf import AdaBeliefOptimizer
import SimpleITK as sitk
import h5py
from scipy.ndimage.interpolation import map_coordinates, zoom
import scipy
from .data_augmentation import random_affine

def set_tf_keys(feed_dict, **kwargs):
    ret = dict([(k + ':0', v) for k, v in feed_dict.items()])
    ret.update([(k + ':0', v) for k, v in kwargs.items()])
    return ret


def masked_mean(arr, mask):
    return tf.reduce_sum(arr * mask) / (tf.reduce_sum(mask) + 1e-9)


class FrameworkUnsupervised:
    net_args = {'class': RecursiveCascadedNetworks}
    framework_name = 'gaffdfrm'

    def __init__(self, devices, image_size, segmentation_class_value, validation=False, fast_reconstruction=False):
        network_class = self.net_args.get('class', RecursiveCascadedNetworks)
        self.summaryType = self.net_args.pop('summary', 'basic')
        self.image_size = image_size

        self.reconstruction = Fast3DTransformer() if fast_reconstruction else Dense3DSpatialTransformer()

        # input place holder
        imgT1_fixed = tf.placeholder(dtype=tf.float32, shape=[
                              None]+image_size+[1], name='voxelT1')
        imgT1_float = tf.placeholder(dtype=tf.float32, shape=[
                              None]+image_size+[1], name='atlasT1')

        seg1 = tf.placeholder(dtype=tf.float32, shape=[
                              None]+image_size+[1], name='seg1')
        seg2 = tf.placeholder(dtype=tf.float32, shape=[
                              None]+image_size+[1], name='seg2')
        point1 = tf.placeholder(dtype=tf.float32, shape=[
                                None, None, 3], name='point1')
        point2 = tf.placeholder(dtype=tf.float32, shape=[
                                None, None, 3], name='point2')#task2

        pseudo_label = tf.placeholder(dtype=tf.float32, shape=[
                              None]+image_size+[1], name='pseudo_label')

        bs = tf.shape(imgT1_fixed)[0]
        Img1, augImg2= imgT1_fixed/255 , imgT1_float/255 #/255

        aug = self.net_args.pop('augmentation', None)
        if aug is None:
            imgs = imgT1_fixed.shape.as_list()[1:4]
            control_fields_1 = transform.sample_power(
                -0.4, 0.4, 3, tf.stack([bs, 5, 5, 5, 3])) * (np.array(imgs) // 4)
            augFlow_1 = transform.free_form_fields(imgs, control_fields_1)


            def augmentation(x,flow):
                if not tflearn.get_training_mode():
                    print('evaluate!!')
                return tf.cond(tflearn.get_training_mode(), lambda: self.reconstruction([x, flow]),
                               lambda: x)

            def augmenetation_pts(incoming,flow):
                def aug(incoming,flow):
                    aug_pt = tf.cast(transform.warp_points(
                        flow, incoming), tf.float32)
                    pt_mask = tf.cast(tf.reduce_all(
                        incoming >= 0, axis=-1, keep_dims=True), tf.float32)
                    return aug_pt * pt_mask - (1 - pt_mask)
                return tf.cond(tflearn.get_training_mode(), lambda: aug(incoming,flow), lambda: incoming)
            
            '''augAtlasT1 = augmentation(preprocessedAtlasT1)
            augAtlasT2 = augmentation(preprocessedAtlasT2)
            augSeg2 = augmentation(seg2)
            augPt2 = augmenetation_pts(point2)'''
            Img1 = augmentation(Img1,augFlow_1)
            seg1 = augmentation(seg1,augFlow_1)
            point1 = augmenetation_pts(point1,augFlow_1)
            pseudo_label = augmentation(pseudo_label,augFlow_1)

            augImg2 = augImg2
            augSeg2 = seg2
            augPt2 = point2
        elif aug == 'identity':
            augFlow = tf.zeros(
                tf.stack([tf.shape(imgT1_fixed)[0], image_size[0], image_size[1], image_size[2], 3]), dtype=tf.float32)
            augSeg2 = seg2
            augPt2 = point2
        else:
            raise NotImplementedError('Augmentation {}'.format(aug))

        learningRate = tf.placeholder(tf.float32, [], 'learningRate')
        if not validation:
            adamOptimizer = tf.train.AdamOptimizer(learningRate)#AdaBeliefOptimizer(learning_rate=learningRate, epsilon=1e-14, rectify=False)
        self.segmentation_class_value = segmentation_class_value
        scheme = self.net_args.pop('scheme', None)
        self.network = network_class(
            self.framework_name, framework=self, fast_reconstruction=fast_reconstruction, scheme=scheme, **self.net_args)
        net_pls = [Img1,augImg2,seg1, augSeg2, point1, augPt2, pseudo_label]
        if devices == 0:
            with tf.device("/cpu:0"):
                self.predictions = self.network(*net_pls)
                if not validation:
                    var_segment =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gaffdfrm/seg_stem")
                    self.adamOpt = adamOptimizer.minimize(
                        self.predictions["loss"])
        else:
            gpus = MultiGPUs(devices)
            if validation:
                self.predictions = gpus(self.network, net_pls, scheme=scheme)
            else:
                self.predictions, self.adamOpt = gpus(
                    self.network, net_pls, opt=adamOptimizer, scheme=scheme)
        self.build_summary(self.predictions)

    @property
    def data_args(self):
        return self.network.data_args

    def build_summary(self, predictions):
        self.loss = tf.reduce_mean(predictions['loss'])
        for k in predictions:
            if k.find('loss') != -1:
                tf.summary.scalar(k, tf.reduce_mean(predictions[k]))
        self.summaryOp = tf.summary.merge_all()

        if self.summaryType == 'full':
            print('@@@@@@')
            tf.summary.scalar('dice_score1', tf.reduce_mean(
                self.predictions['dice_score1']))
            tf.summary.scalar('landmark_dist', masked_mean(
                self.predictions['landmark_dist'], self.predictions['pt_mask']))
            preds = tf.reduce_sum(
                tf.cast(self.predictions['jacc_score'] > 0, tf.float32))
            tf.summary.scalar('jacc_score', tf.reduce_sum(
                self.predictions['jacc_score']) / (preds + 1e-8))
            
            self.summaryExtra = tf.summary.merge_all()
        else:
            self.summaryExtra = self.summaryOp
        self.summaryImages1 = tf.summary.image('fixed_img', tf.reshape(self.predictions['image_fixed_T1'][:,96,:,:,0], (1,128,128,1)))
        self.summaryImages2 = tf.summary.image('warped_moving_img', tf.reshape(self.predictions['warped_moving_T1'][:,96,:,:,0], (1,128,128,1)))
        self.summaryImages3 = tf.summary.image('image_float_T1', tf.reshape(self.predictions['image_float_T1'][:,96,:,:,0], (1,128,128,1)))
        self.summaryImages = tf.summary.merge([self.summaryImages1,self.summaryImages2,self.summaryImages3])

    def get_predictions(self, *keys):
        return dict([(k, self.predictions[k]) for k in keys])

    def validate_clean(self, sess, generator, keys=None):
        for fd in generator:
            _ = fd.pop('id1')
            _ = fd.pop('id2')
            _ = sess.run(self.get_predictions(*keys),
                         feed_dict=set_tf_keys(fd))
    def fusion_dices(self,candidates, target_seg):
        def compute_dice_coefficient(mask_gt, mask_pred):
            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return 0
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2*volume_intersect / volume_sum
        fusion_label_onehot = np.mean(candidates,axis=0,keepdims=False)
        fusion_label = np.argmax(fusion_label_onehot,axis=-1)
        dices = []
        for i in range(fusion_label_onehot.shape[-1]):
            dices.append(compute_dice_coefficient(target_seg==i,fusion_label==i))
        return np.array(dices)[1:]

    def validate(self, sess, generator, keys=None, summary=False, predict=False, show_tqdm=False):
        if keys is None:
            keys = ['dice_score1','dice_score2', 'landmark_dist', 'pt_mask', 'jacc_score']
            # if self.segmentation_class_value is not None:
            #     for k in self.segmentation_class_value:
            #         keys.append('jacc_{}'.format(k))
        full_results = dict([(k, list()) for k in keys])
        if not summary:
            full_results['id1'] = []
            full_results['id2'] = []
            if predict:
                full_results['seg1'] = []
                full_results['seg2'] = []
                full_results['imgT1_fixed'] = []
                full_results['imgT1_float'] = []
        tflearn.is_training(False, sess)
        if show_tqdm:
            generator = tqdm(generator)
        i = 0
        for FD in generator:
            i += 1
            
            '''if (i>1):
                break
            '''
            if isinstance(FD, list):
                if 'id1' not in FD[0]:
                    break
                keys.append("warped_seg_moving")
                candidates = []
                for fd in FD:
                    id1 = fd.pop("id1")
                    id2 = fd.pop('id2')
                    #print(id2,id1)
                    results = sess.run(self.get_predictions(
                        *keys), feed_dict=set_tf_keys(fd))
                    candidates.append(np.squeeze(results.pop("warped_seg_moving"))/255.0)
                results["dices"] = np.expand_dims(self.fusion_dices(candidates, np.squeeze(fd['seg1'])),0)
                results["dice_score1"] = np.expand_dims(np.mean(results["dices"]),0)
            else:
                fd = FD
                if 'id1' not in fd:
                    break
                id1 = fd.pop('id1')
                id2 = fd.pop('id2')
                #print(id1,id2)
                results = sess.run(self.get_predictions(
                    *keys), feed_dict=set_tf_keys(fd))
            if not summary:
                results['id1'] = id1
                results['id2'] = id2
                if predict:
                    results['seg1'] = fd['seg1']
                    results['seg2'] = fd['seg2']
                    results['imgT1_fixed'] = fd['voxelT1']
                    results['imgT1_float'] = fd['atlasT1']

            mask = np.where([i and j for i, j in zip(id1, id2)])
            for k, v in results.items():
                if k not in full_results:
                    continue
                full_results[k].append(v[mask])
        if 'landmark_dist' in full_results and 'pt_mask' in full_results:
            pt_mask = full_results.pop('pt_mask')
            full_results['landmark_dist'] = [arr * mask for arr,
                                             mask in zip(full_results['landmark_dist'], pt_mask)]
        for k in full_results:
            #print(k)
            #print(np.array(full_results[k]).shape)
            full_results[k] = np.concatenate(full_results[k], axis=0)
            if k == 'dices' or k == 'dices2' or k == 'dices3' or k == 'dices_fusion':
                print(k,': ', np.mean(full_results[k], axis=0))
            if summary:
                full_results[k] = full_results[k].mean()

        return full_results

    def validate2(self, sess, generator, keys=None, summary=False, predict=False, show_tqdm=False):
        def jacobian_determinant(disp):
            _, _, H, W, D = disp.shape
            
            gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
            grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
            gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

            gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                                scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                                scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
            
            grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                                scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                                scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
            
            gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                                scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                                scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

            grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

            jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
            jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
            jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
                    jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
                    jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
                
            return jacdet
        file = h5py.File('/home/wangsheng/my_project/OS_seg/datasets/L2Reg2.h5','r')
        def compute_dice_coefficient(seg1, seg2):
            return 2*(seg1&seg2).sum()/(seg1.sum()+seg2.sum())
        def metrices(id1, id2, realflow):
            id1 = str(id1[0])
            id2 = str(id2[0])
            id_fixed =  id1.replace('L2Reg3_','')
            id_moving = id2.replace('L2Reg3_','')

            fixed = np.array(file[str(id_fixed)]['segmentation'])
            moving = np.array(file[str(id_moving)]['segmentation'])
            disp_field = 2*np.array([zoom(realflow[...,i], 2, order=2) for i in range(3)])
            jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(0.000000001, 1000000000)
            log_jac_det = np.log(jac_det).std()

            D, H, W = disp_field.shape[1:]
            identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
            moving_warped = map_coordinates(moving, identity + disp_field, order=0)
            dice = 0
            count = 0
            for i in range(1, 36):
                if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
                    continue
                dice += compute_dice_coefficient((fixed==i), (moving_warped==i))
                count += 1
            dice /= count
            return dice, log_jac_det
        
        if keys is None:
            keys = ['dice_score1','dice_score2', 'landmark_dist', 'pt_mask', 'jacc_score']
            # if self.segmentation_class_value is not None:
            #     for k in self.segmentation_class_value:
            #         keys.append('jacc_{}'.format(k))
        keys.append('real_flow')
        full_results = dict([(k, list()) for k in keys])
        full_results['jacb'] = []
        if not summary:
            full_results['id1'] = []
            full_results['id2'] = []
            if predict:
                full_results['seg1'] = []
                full_results['seg2'] = []
                full_results['imgT1_fixed'] = []
                full_results['imgT1_float'] = []
        tflearn.is_training(False, sess)
        if show_tqdm:
            generator = tqdm(generator)
        i = 0

        for fd in generator:
            id1 = fd.pop('id1')
            id2 = fd.pop('id2')
            #print(id1,id2)

            if id1 =='':
                break
            results = sess.run(self.get_predictions(
                *keys), feed_dict=set_tf_keys(fd))
            dice, jacb = metrices(id1, id2, np.squeeze(results['real_flow']))

            results['dice_score1'] = np.reshape(dice, results['dice_score1'].shape)
            results['jacb'] = np.reshape(jacb, results['dice_score1'].shape)
            if not summary:
                results['id1'] = id1
                results['id2'] = id2
                if predict:
                    results['seg1'] = fd['seg1']
                    results['seg2'] = fd['seg2']
                    results['imgT1_fixed'] = fd['voxelT1']
                    results['imgT1_float'] = fd['atlasT1']

            mask = np.where([i and j for i, j in zip(id1, id2)])
            for k, v in results.items():
                full_results[k].append(v[mask])
        if 'landmark_dist' in full_results and 'pt_mask' in full_results:
            pt_mask = full_results.pop('pt_mask')
            full_results['landmark_dist'] = [arr * mask for arr,
                                             mask in zip(full_results['landmark_dist'], pt_mask)]
        for k in full_results:
            #print(k)
            #print(np.array(full_results[k]).shape)
            full_results[k] = np.concatenate(full_results[k], axis=0)
            if k == 'dices' or k == 'dices2' or k == 'dices3' or k == 'dices_fusion':
                print(k,': ', np.mean(full_results[k], axis=0))
            if summary:
                full_results[k] = full_results[k].mean()

        return full_results
        
    def validate2_task2(self, sess, generator, keys=None, summary=False, predict=False, show_tqdm=False):
        def compute_tre(x, y, spacing):
            return np.linalg.norm((x - y) * spacing, axis=1)
        def jacobian_determinant(disp):
            _, _, H, W, D = disp.shape
            
            gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
            grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
            gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

            gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                                scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                                scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
            
            grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                                scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                                scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
            
            gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                                scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                                scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

            grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

            jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
            jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
            jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
                    jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
                    jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
                
            return jacdet
        file = h5py.File('/home/wangsheng/my_project/OS_seg/datasets/L2Reg_task2.h5','r')
        
        def metrices(id1, id2, realflow):
            id1 = str(id1[0])
            id2 = str(id2[0])
            id_fixed =  id1.replace('L2Reg_task2_','')+'_full'
            id_moving = id2.replace('L2Reg_task2_','')+'_full'
            spacing = (1.75, 1.25, 1.75)

            lms_fixed = np.array(file[str(id_fixed)]['point']).astype(np.float32)
            lms_moving = np.array(file[str(id_moving)]['point']).astype(np.float32)
            img = np.array(file[str(id_fixed)]['volumeT1'])
            
            disp_field = 2.0*np.array([zoom(realflow[...,i], 2, order=2) for i in range(3)])
            jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(0.000000001, 1000000000)
            log_jac_det = np.log(jac_det)
            log_jac_det = np.ma.MaskedArray(log_jac_det, 1-img[2:-2, 2:-2, 2:-2]).std()

            lms_fixed_disp_x = map_coordinates(disp_field[0], lms_fixed.transpose())
            lms_fixed_disp_y = map_coordinates(disp_field[1], lms_fixed.transpose())
            lms_fixed_disp_z = map_coordinates(disp_field[2], lms_fixed.transpose())
            lms_fixed_disp = np.array((lms_fixed_disp_x, lms_fixed_disp_y, lms_fixed_disp_z)).transpose()
            lms_fixed_warped = lms_fixed + lms_fixed_disp
            tre = compute_tre(lms_fixed_warped, lms_moving, spacing)
            origin_tre = compute_tre(lms_fixed, lms_moving, spacing)
            '''print(tre.shape)
            with open('shit.txt','a') as txt:
                for i in range(origin_tre.shape[0]):
                    txt.writelines(str(tre[i])+'\n')
                txt.writelines('\n')
            '''

            D, H, W = disp_field.shape[1:]
            identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
            moving = np.array(file[str(id_moving)]['volumeT1']).astype(np.float32)
            moving_warped = map_coordinates(moving, identity + disp_field, order=2)

            return tre.mean(), log_jac_det, origin_tre.mean()


        if keys is None:
            keys = ['dice_score1','dice_score2', 'landmark_dist', 'pt_mask', 'jacc_score']
            # if self.segmentation_class_value is not None:
            #     for k in self.segmentation_class_value:
            #         keys.append('jacc_{}'.format(k))
        keys.append('real_flow')
        full_results = dict([(k, list()) for k in keys])
        full_results['jacb'] = []
        full_results['TRE'] = []
        if not summary:
            full_results['id1'] = []
            full_results['id2'] = []
            if predict:
                full_results['seg1'] = []
                full_results['seg2'] = []
                full_results['imgT1_fixed'] = []
                full_results['imgT1_float'] = []
        tflearn.is_training(False, sess)
        if show_tqdm:
            generator = tqdm(generator)
        i = 0

        for fd in generator:
            id1 = fd.pop('id1')
            id2 = fd.pop('id2')
            #print(id1,id2)

            if id1 =='':
                break
            results = sess.run(self.get_predictions(
                *keys), feed_dict=set_tf_keys(fd))
            tre, jacb, origin_tre = metrices(id1, id2, np.squeeze(results['real_flow']))
            print(tre, origin_tre)

            results['TRE'] = np.reshape(tre, results['dice_score1'].shape)
            results['jacb'] = np.reshape(jacb, results['dice_score1'].shape)
            if not summary:
                results['id1'] = id1
                results['id2'] = id2
                if predict:
                    results['seg1'] = fd['seg1']
                    results['seg2'] = fd['seg2']
                    results['imgT1_fixed'] = fd['voxelT1']
                    results['imgT1_float'] = fd['atlasT1']

            mask = np.where([i and j for i, j in zip(id1, id2)])
            for k, v in results.items():
                full_results[k].append(v[mask])
        if 'landmark_dist' in full_results and 'pt_mask' in full_results:
            pt_mask = full_results.pop('pt_mask')
            full_results['landmark_dist'] = [arr * mask for arr,
                                             mask in zip(full_results['landmark_dist'], pt_mask)]
        for k in full_results:
            #print(k)
            #print(np.array(full_results[k]).shape)
            full_results[k] = np.concatenate(full_results[k], axis=0)
            if k == 'dices' or k == 'dices2' or k == 'dices3' or k == 'dices_fusion':
                print(k,': ', np.mean(full_results[k], axis=0))
            if summary:
                full_results[k] = full_results[k].mean()

        return full_results
