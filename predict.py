
import argparse
import os
import json
import re
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from PIL import Image
import math
import scipy.misc
import xlwt
from metric import get_metrices

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, default=None,
                    help='Specifies a previous checkpoint to load')
parser.add_argument('-r', '--rep', type=int, default=1,
                    help='Number of times of shared-weight cascading')
parser.add_argument('-g', '--gpu', type=str, default='0',
                    help='Specifies gpu device(s)')
parser.add_argument('-d', '--dataset', type=str, default='datasets/brain.json',
                    help='Specifies a data config')
parser.add_argument('-v', '--val_subset', type=str, default=None)
parser.add_argument('--batch', type=int, default=1, help='Size of minibatch')
parser.add_argument('--fast_reconstruction', action='store_true')
parser.add_argument('--paired', action='store_true')
parser.add_argument('--data_args', type=str, default=None)
parser.add_argument('--net_args', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--scheme', type=str, default=None, help='chose reg、seg、reg_supervise')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import tensorflow as tf
import tflearn

import network
import data_util.liver
import data_util.brain


def main():
    if args.checkpoint is None:
        print('Checkpoint must be specified!')
        return
    if ':' in args.checkpoint:
        args.checkpoint, steps = args.checkpoint.split(':')
        steps = int(steps)
    else:
        steps = None
    args.checkpoint = find_checkpoint_step(args.checkpoint, steps)
    print(args.checkpoint)
    model_dir = os.path.dirname(args.checkpoint)
    try:
        with open(os.path.join(model_dir, 'args.json'), 'r') as f:
            model_args = json.load(f)
        print(model_args)
    except Exception as e:
        print(e)
        model_args = {}

    if args.dataset is None:
        args.dataset = model_args['dataset']
    if args.data_args is None:
        args.data_args = model_args['data_args']

    Framework = network.FrameworkUnsupervised
    Framework.net_args['base_network'] = model_args['base_network']
    Framework.net_args['n_cascades'] = model_args['n_cascades']
    Framework.net_args['rep'] = args.rep
    Framework.net_args['augmentation'] = 'identity'
    Framework.net_args['scheme'] = args.scheme
    Framework.net_args.update(eval('dict({})'.format(model_args['net_args'])))
    if args.net_args is not None:
        Framework.net_args.update(eval('dict({})'.format(args.net_args)))
    with open(os.path.join(args.dataset), 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [160, 160, 160])
        image_type = cfg.get('image_type')
    gpus = 0 if args.gpu == '-1' else len(args.gpu.split(','))
    framework = Framework(devices=gpus, image_size=image_size, segmentation_class_value=cfg.get(
        'segmentation_class_value', None), fast_reconstruction=args.fast_reconstruction, validation=True)
    print('Graph built')

    Dataset = eval('data_util.{}.Dataset'.format(image_type))
    ds = Dataset(args.dataset, batch_size=args.batch, paired=args.paired, **
                 eval('dict({})'.format(args.data_args)))
                 
    config = tf.ConfigProto(allow_soft_placement = True) 
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    tf.global_variables_initializer().run(session=sess)

    saver = tf.train.Saver(tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES))
    checkpoint = args.checkpoint
    def optimistic_restore(session, save_file):
                reader = tf.train.NewCheckpointReader(save_file)
                saved_shapes = reader.get_variable_to_shape_map()
                var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                                if var.name.split(':')[0] in saved_shapes])
                restore_vars = []
                name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
                with tf.variable_scope('', reuse=True):
                    for var_name, saved_var_name in var_names:
                        curr_var = name2var[saved_var_name]
                        var_shape = curr_var.get_shape().as_list()
                        if var_shape == saved_shapes[saved_var_name]:
                            restore_vars.append(curr_var)
                saver = tf.train.Saver(restore_vars)
                saver.restore(session, save_file)
                
    def dense_restore(session, save_file):
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        var_list = {}
        def name_convert(Str):
            b = Str.find('stem_0_')
            e = Str.find('/',b)
            Str = Str.replace(Str[b:e],'stem_0')
            b = Str.find('dense_')
            e = Str.find('/',b)
            num_o = int(Str[b+6:e])
            num_n = num_o%6
            if num_n == 0:
                num_n = 6
            new_Str = Str.replace(str(num_o),str(num_n))
            return new_Str
        for var_name in saved_shapes:
            fuck = 0
            #print(var_name)
        for var in tf.global_variables():
            if var.name.split(':')[0] not in saved_shapes:
                print(var)
                if 'deform' in var.name:
                    var_list[name_convert(var.name.split(':')[0])]=var
                    print('convert %s to %s'%(var.name, name_convert(var.name.split(':')[0])))
        if len(var_list) != 0:
            print('dense restored!!!')
            saver_dense = tf.train.Saver(var_list)
            saver_dense.restore(session, save_file)
            
    #saver.restore(sess, checkpoint)
    optimistic_restore(sess, checkpoint)
    dense_restore(sess, checkpoint)
    ############feature net##############
    var_feature = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gaffdfrm/feature')
    print(var_feature)
    var_list = dict(zip(map(lambda x:x.name.replace('feature','deform_stem_0').split(':')[0], var_feature), var_feature))
    for var in var_list:
        print(var)
    #saver_feature = tf.train.Saver(var_list)
    #saver_feature.restore(sess, checkpoint)
    tflearn.is_training(False, session=sess)
    ############feature net##############

    val_subsets = [data_util.liver.Split.VALID]
    if args.val_subset is not None:
        val_subsets = args.val_subset.split(',')

    tflearn.is_training(False, session=sess)
    if args.scheme == "seg":
        seg_key = "seg_result"
    else:
        seg_key = "warped_seg_moving"
    keys = ['image_fixed_T1', 'seg1', 'warped_moving_T1', seg_key,'real_flow']
    if not os.path.exists('evaluate'):
        os.mkdir('evaluate')
    path_prefix = os.path.join('evaluate', short_name(checkpoint))
    if args.rep > 1:
        path_prefix = path_prefix + '-rep' + str(args.rep)
    if args.name is not None:
        path_prefix = path_prefix + '-' + args.name
    for val_subset in val_subsets:
        if args.val_subset is not None:
            output_fname = path_prefix + '-' + str(val_subset) + '.txt'
        else:
            output_fname = path_prefix + '.txt'
        with open(output_fname, 'w') as fo:
            print("Validation subset {}".format(val_subset))
            gen = ds.generator(val_subset, loop=False)
            results = framework.validate(sess, gen, keys=keys, summary=False, predict=True, show_tqdm=True) 
            ##################image save#########################
            image_save_path = os.path.join('./test_images', short_name(checkpoint))
            if not os.path.isdir(image_save_path):
                os.makedirs(image_save_path)
            
            for i in range(len(results['id1'])):
                sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(results['seg1'][i][:,:,:,0])), image_save_path+'/'+results['id1'][i]+'_seg.nii', True)
                sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(results['image_fixed_T1'][i][:,:,:,0])), image_save_path+'/'+results['id1'][i]+'_T1_img.nii', True)
                
                if args.scheme == 'seg':
                    warped_seg = np.squeeze(np.zeros(results['seg_result'][i][:,:,:,0].shape))
                    for seg in range(results['seg_result'][i].shape[-1]):
                        sub_warp = np.squeeze((results['seg_result'][i])[:,:,:,seg])
                        sub_warp = np.where(sub_warp>127.5,seg,0)
                        warped_seg += sub_warp
                    sitk.WriteImage(sitk.GetImageFromArray(warped_seg), image_save_path+'/'+results['id1'][i]+'_seg_result.nii', True)
                else:
                    warped_seg = np.squeeze(np.zeros(results['warped_seg_moving'][i][:,:,:,0].shape))
                    for seg in range(results['warped_seg_moving'][i].shape[-1]):
                        sub_warp = np.squeeze((results['warped_seg_moving'][i])[:,:,:,seg])
                        sub_warp = np.where(sub_warp>127.5,seg,0)
                        warped_seg += sub_warp
                    sitk.WriteImage(sitk.GetImageFromArray(warped_seg), image_save_path+'/'+results['id1'][i]+'_warped_seg.nii', True)
    
            workbook = xlwt.Workbook()
            sheet = workbook.add_sheet('result')
            keys_xls = ["img_id","hd95_score","dice_score","LogJacStd_score","mae_score","ncc_score","hd95s","dices"]
            full_results = dict([(k, list()) for k in keys_xls])
            for i in range(len(results['id1'])):
                warped_seg = np.squeeze(np.zeros(results[keys[-2]][i][:,:,:,0].shape))
                labels = []
                for num,label in enumerate(cfg["segmentation_class_value"].values()):
                    sub_warp = np.squeeze((results[seg_key][i])[:,:,:,num])
                    sub_warp = np.where(sub_warp>127.5,label,0)
                    if(label !=0):
                        labels.append(label)
                    warped_seg += sub_warp
                metirc_results  = get_metrices(
                                                np.squeeze(results['image_fixed_T1'][i][:,:,:,0]),
                                                np.squeeze(results['warped_moving_T1'][i][:,:,:,0]), 
                                                np.squeeze(results['seg1'][i][:,:,:,0]), 
                                                warped_seg, 
                                                np.transpose(np.squeeze(results['real_flow'][i]),(3,0,1,2)), 
                                                labels)
                for k, v in metirc_results.items():
                    full_results[k].append(v)
                full_results["img_id"].append(results['id1'][i])
            for key in full_results:
                full_results[key] = np.array(full_results[key])
            j = 0
            for key in full_results:
                if "score" in key or "id" in key:
                    sheet.write(0, j, key)
                    for i in range(full_results[key].shape[0]):
                        if "score" in key:
                            sheet.write(i+1, j, float(full_results[key][i]))
                        else:
                            sheet.write(i+1, j, full_results[key][i])
                    if "score" in key:
                        sheet.write(i+2, j, float(np.mean(full_results[key])))
                    j+=1
                else:
                    for s,seg in enumerate(labels):
                        sheet.write(0, j, key+'_{}'.format(seg))
                        for i in range(full_results[key].shape[0]):
                            sheet.write(i+1, j, float(full_results[key][i,s]))
                        sheet.write(i+2, j, float(np.mean(full_results[key][:,s])))
                        j+=1
            workbook.save(path_prefix+'_metrices.xls')
            
def cbimage(img1, img2):
    shape = img1.shape
    num = 20
    grid = np.zeros(shape)
    bg_p_x = [int(shape[0]/num*i) for i in range(num)]
    bg_p_y = [int(shape[1]/num*i) for i in range(num)]
    for i in range(0, num, 2):
        for j in range(0, num, 2):
            grid[bg_p_x[i]:bg_p_x[i]+shape[0]//num,bg_p_y[j]:bg_p_y[j]+shape[1]//num] = 1
            grid[bg_p_x[i+1]:bg_p_x[i+1]+shape[0]//num,bg_p_y[j+1]:bg_p_y[j+1]+shape[1]//num] = 1
    img1_grid = img1*grid
    img2_grid = img2*(1-grid)
    cbimage = img1_grid+img2_grid
    return cbimage

def write_excel_xls(path, sheet_name, value):
    index = len(value) 
    workbook = xlwt.Workbook() 
    sheet = workbook.add_sheet(sheet_name)  
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j]) 
    workbook.save(path) 
     
def short_name(checkpoint):
    cpath, steps = os.path.split(checkpoint)
    _, exp = os.path.split(cpath)
    return exp + '-' + steps

def RenderFlow(flow, coef = 15, channel = (0, 1, 2), thresh = 1):
    flow = flow[:, :, 64]
    im_flow = np.stack([flow[:, :, c] for c in channel], axis = -1)
    return im_flow

def find_checkpoint_step(checkpoint_path, target_steps=None):
    pattern = re.compile(r'model-(\d+).index')
    checkpoints = []
    for f in os.listdir(checkpoint_path):
        m = pattern.match(f)
        if m:
            steps = int(m.group(1))
            checkpoints.append((-steps if target_steps is None else abs(
                target_steps - steps), os.path.join(checkpoint_path, f.replace('.index', ''))))
    return min(checkpoints, key=lambda x: x[0])[1]

if __name__ == '__main__':
    main()
