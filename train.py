#####REG+SEG#####
import argparse
import numpy as np
import os
import json
import h5py
import copy
import collections
import re
import datetime
import hashlib
import time
from timeit import default_timer

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--base_network', type=str, default='RWUNET_v1',
                    help='Specifies the base network (either VTN or VoxelMorph)')
parser.add_argument('-n', '--n_cascades', type=int, default=1,
                    help='Number of cascades')
parser.add_argument('-r', '--rep', type=int, default=1,
                    help='Number of times of shared-weight cascading')
parser.add_argument('-g', '--gpu', type=str, default='1',
                    help='Specifies gpu device(s)')
parser.add_argument('-c', '--checkpoint', type=str, default=None,##Apr04-0957
                    help='Specifies a previous checkpoint to start with')
parser.add_argument('-d', '--dataset', type=str, default="datasets/brain.json",
                    help='Specifies a data config')
parser.add_argument('--batch', type=int, default=1,
                    help='Number of image pairs per batch')
parser.add_argument('--round', type=int, default=2000,
                    help='Number of batches per epoch')
parser.add_argument('--epochs', type=float, default=10,
                    help='Number of epochs')
parser.add_argument('--fast_reconstruction', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--val_steps', type=int, default=200)
parser.add_argument('--net_args', type=str, default='')
parser.add_argument('--data_args', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--clear_steps', action='store_true')
parser.add_argument('--finetune', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--logs', type=str, default='')
parser.add_argument('--scheme', type=str, default='reg', help='chose reg、seg、reg_supervise')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import tensorflow as tf
import tflearn
import keras

import network
import data_util.liver
import data_util.brain
from data_util.data import Split

def main():

    #np.random.seed(1234)
    #tf.set_random_seed(1234)
    repoRoot = os.path.dirname(os.path.realpath(__file__))
    print('repoRoot:',repoRoot)

    if args.finetune is not None:
        args.clear_steps = True

    batchSize = args.batch
    iterationSize = args.round

    gpus = 0 if args.gpu == '-1' else len(args.gpu.split(','))

    Framework = network.FrameworkUnsupervised
    Framework.net_args['base_network'] = args.base_network
    Framework.net_args['n_cascades'] = args.n_cascades
    Framework.net_args['rep'] = args.rep
    if args.scheme == "seg":
        Framework.net_args['augmentation'] = "identity"
    else:
        Framework.net_args['augmentation'] = None
    Framework.net_args['scheme'] = args.scheme
    Framework.net_args.update(eval('dict({})'.format(args.net_args)))
    with open(os.path.join(args.dataset), 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [160, 160, 160])
        image_type = cfg.get('image_type')
    framework = Framework(devices=gpus, image_size=image_size, segmentation_class_value=cfg.get('segmentation_class_value', None), fast_reconstruction = args.fast_reconstruction)
    Dataset = eval('data_util.{}.Dataset'.format(image_type))
    print('Graph built.')

    # load training set and validation set

    def set_tf_keys(feed_dict, **kwargs):
        ret = dict([(k + ':0', v) for k, v in feed_dict.items()])
        ret.update([(k + ':0', v) for k, v in kwargs.items()])
        return ret

    config = tf.ConfigProto(allow_soft_placement = True) 
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES), keep_checkpoint_every_n_hours=5)
        if args.checkpoint is None:
            steps = 0
            tf.global_variables_initializer().run()
        else:
            if '\\' not in args.checkpoint and '/' not in args.checkpoint:
                args.checkpoint = os.path.join(
                    repoRoot, 'weights', args.checkpoint)
            if os.path.isdir(args.checkpoint):
                print('args.checkpoint: ', args.checkpoint)
                args.checkpoint = tf.train.latest_checkpoint(args.checkpoint)

            tf.compat.v1.global_variables_initializer().run()
            
            checkpoints = args.checkpoint.split(';')

            if args.clear_steps:
                steps = 0
            else:
                steps = int(re.search('model-(\d+)', checkpoints[0]).group(1))

            #加载参数
            def optimistic_restore(session, save_file):
                reader = tf.train.NewCheckpointReader(save_file)
                saved_shapes = reader.get_variable_to_shape_map()
                print( 'tf.global_variables(): ',tf.global_variables())
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

            for cp in checkpoints:
                optimistic_restore(sess, cp)
                print(cp)
                for var in tf.global_variables():
                    #if 'deform' in var.name:
                    print('var: ',var)
                
                
                var_feature = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gaffdfrm/feature')
                print(var_feature)
                var_list = dict(zip(map(lambda x:x.name.replace('feature','deform_stem_0').split(':')[0], var_feature), var_feature))
                saver_feature = tf.train.Saver(var_list)
                saver_feature.restore(sess, cp)
                
                
                # var_feature = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gaffdfrm/deform_teacher')
                # var_list = dict(zip(map(lambda x:x.name.replace('deform_teacher','deform_stem_0').split(':')[0], var_feature), var_feature))
                # saver_feature = tf.train.Saver(var_list)
                # saver_feature.restore(sess, cp)
                
                #var_seg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gaffdfrm/seg_stem')
                #saver_deform_2 = tf.train.Saver(var_list = var_seg )
                #saver_deform_2.restore(sess, cp)

                #var_deform = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gaffdfrm/deform_stem_0')
                #saver_deform_1 = tf.train.Saver(var_list = var_deform)
                #saver_deform_1.restore(sess, 'weights/Dec04-1332/model-18200')

                #var_seg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gaffdfrm/seg_stem')
                #saver_deform_2 = tf.train.Saver(var_list = var_seg )
                #saver_deform_2.restore(sess, '/home/wangsheng/my_project/OS_seg/weights/Jun20-1314/model-1500')


        data_args = eval('dict({})'.format(args.data_args))
        data_args.update(framework.data_args)
        print('data_args', data_args)
        dataset = Dataset(args.dataset, **data_args)
        if args.finetune is not None:
            if 'finetune-train-%s' % args.finetune in dataset.schemes:
                dataset.schemes[Split.TRAIN] = dataset.schemes['finetune-train-%s' %
                                                               args.finetune]
            if 'finetune-val-%s' % args.finetune in dataset.schemes:
                dataset.schemes[Split.VALID] = dataset.schemes['finetune-val-%s' %
                                                               args.finetune]
            print('train', dataset.schemes[Split.TRAIN])
            print('val', dataset.schemes[Split.VALID])
        if args.scheme == 'seg':
            if_seg=True
        else:
            if_seg=False
        generator = dataset.generator(Split.TRAIN, batch_size=batchSize, loop=True, pair_train=False,  if_seg=if_seg)

        if not args.debug:
            if args.finetune is not None:
                run_id = os.path.basename(os.path.dirname(args.checkpoint))
                if not run_id.endswith('_ft' + args.finetune):
                    run_id = run_id + '_ft' + args.finetune
            else:
                pad = ''
                retry = 1
                while True:
                    dt = datetime.datetime.now(
                        tz=datetime.timezone(datetime.timedelta(hours=8)))
                    run_id = dt.strftime('%b%d-%H%M') + pad
                    modelPrefix = os.path.join(repoRoot, 'weights', run_id)
                    try:
                        os.makedirs(modelPrefix)
                        break
                    except Exception as e:
                        print('Conflict with {}! Retry...'.format(run_id))
                        pad = '_{}'.format(retry)
                        retry += 1
            modelPrefix = os.path.join(repoRoot, 'weights', run_id)
            if not os.path.exists(modelPrefix):
                os.makedirs(modelPrefix)
            if args.name is not None:
                run_id += '_' + args.name
            if args.logs is None:
                log_dir = 'logs'
            else:
                log_dir = os.path.join('logs', args.logs)
            summary_path = os.path.join(repoRoot, log_dir, run_id)
            if not os.path.exists(summary_path):
                os.makedirs(summary_path)
            summaryWriter = tf.summary.FileWriter(summary_path, sess.graph)
            with open(os.path.join(modelPrefix, 'args.json'), 'w') as fo:
                json.dump(vars(args), fo)

        if args.finetune is not None:
            learningRates = [1e-5 / 2, 1e-5 / 2, 1e-5 / 2, 1e-5 / 4, 1e-5 / 8]
            #args.epochs = 1
        else:
            #learningRates = [1e-4/4, 1e-4/4, 1e-4/4,1e-4/8, 1e-4 / 8, 1e-4 / 8, 1e-4 / 16, 1e-4 / 16, 1e-4 / 32,1e-4/32]
            learningRates = [1e-4, 1e-4, 1e-4,1e-4, 1e-4 / 2, 1e-4 / 2, 1e-4 / 2, 1e-4 / 4, 1e-4 / 4,1e-4/8]#10 epoch
            #learningRates = [1e-4, 1e-4, 1e-4,1e-4, 1e-4 / 2, 1e-4 / 4, 1e-4 / 8, 1e-4 / 8, 1e-4 / 8]#9 epoch 

            # Training

        def get_lr(steps):
            m = args.lr / learningRates[0]
            return m * learningRates[steps // iterationSize]

        last_save_stamp = time.time()
        best_dice_score = 0.0
        while True:
            if hasattr(framework, 'get_lr'):
                lr = framework.get_lr(steps, batchSize)
            else:
                lr = get_lr(steps)
            t0 = default_timer()
            fd = next(generator)
            print('fd :',fd['voxelT1'].shape)
            fd.pop('mask', [])
            id1 = fd.pop('id1', [])
            id2 = fd.pop('id2', [])
            t1 = default_timer()
            tflearn.is_training(True, session=sess)
            #写入loss,执行优化
            summ, _ = sess.run([framework.summaryExtra, framework.adamOpt],
                               set_tf_keys(fd, learningRate=lr))

            for v in tf.Summary().FromString(summ).value:
                if v.tag == 'loss':
                    loss = v.simple_value

            steps += 1
            if args.debug or steps % 10 == 0:
                if steps >= args.epochs * iterationSize:
                    break

                if not args.debug:
                    summaryWriter.add_summary(summ, steps)

                if steps % 100 == 0:
                    if hasattr(framework, 'summaryImages'):
                        summ, = sess.run([framework.summaryImages],
                                         set_tf_keys(fd))
                        summaryWriter.add_summary(summ, steps)

                if steps % 50 == 0:
                    print('*%s* ' % run_id,
                          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                          'Steps %d, Total time %.2f, data %.2f%%. Loss %.3e lr %.3e' % (steps,
                                                                                         default_timer() - t0,
                                                                                         (t1 - t0) / (
                                                                                             default_timer() - t0),
                                                                                         loss,
                                                                                         lr),
                          end='\n')

                #if time.time() - last_save_stamp > 3600 or steps % iterationSize == iterationSize - 500:
                if steps == args.epochs * iterationSize-500:
                    last_save_stamp = time.time()
                    '''saver.save(sess, os.path.join(modelPrefix, 'model'),
                               global_step=steps, write_meta_graph=True)
                    '''

                if args.debug or steps % args.val_steps == 0:
                    #try:
                    #tflearn.is_training(False, session=sess)
                    val_gen = dataset.generator(
                        Split.VALID, loop=False, batch_size=batchSize,  if_seg= if_seg)
                    if args.scheme == 'reg' or args.scheme== 'reg_supervise':
                        keys = ['dice_score1', 'dices', 'landmark_dist', 'pt_mask', 'jacc_score','ncc_score']
                    else:
                        keys = ['dice_score1', 'dices2', 'dices3', 'dice_score2', 'dices_pseudo']
                        
                    metrics = framework.validate(
                        sess, val_gen,keys=keys, summary=True)

                    val_summ = tf.Summary(value=[
                        tf.Summary.Value(tag='val_' + k, simple_value=v) for k, v in metrics.items()
                    ])
                    if args.scheme == 'reg' or args.scheme == 'reg_supervise':
                        dice_score = metrics['dice_score1']
                    else:
                        dice_score = metrics['dice_score2']
                    print('dice:',dice_score)#if use segnet,change dice_score1 to dice_score2
                    if dice_score>best_dice_score:
                        best_dice_score = dice_score
                        print('saving best dice sore:{}'.format(best_dice_score))
                        saver.save(sess, os.path.join(modelPrefix, 'model'),global_step=steps,write_meta_graph=False)
                        with open(os.path.join(modelPrefix,'log.txt'),'a+') as f:
                            f.write('saving best dice sore:{},steps={} \n'.format(best_dice_score,steps))
                    summaryWriter.add_summary(val_summ, steps)
                    '''except:
                        if steps == args.val_steps:
                            print('Step {}, validation failed!'.format(steps))
                    '''
    print('Finished.')


if __name__ == '__main__':
    main()
