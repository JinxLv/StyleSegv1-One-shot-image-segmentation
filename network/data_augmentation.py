import tensorflow as tf
import numpy as np
import math
import os
import SimpleITK as sitk
import sys
import h5py
import random
from .spatial_transformer import Dense3DSpatialTransformer
from tensorflow.python.framework import ops as _ops
from tensorflow.python.ops import manip_ops
def random_affine(img,
                  scale_range = [0.9,1.2,0.9,1.2,1,1],#0.9,1.1
                  degree_range = [-0.2, 0.2, -0.2, 0.2, -0.0, 0.0],#0.3
                  translaton_range = [-0,0,-0,0,-2,2],#3
                  sx=None, sy=None, sz=None):
    
    #scale param
    s_x = tf.random_uniform([],minval = scale_range[0],maxval = scale_range[1])
    s_y = tf.random_uniform([],minval = scale_range[2],maxval = scale_range[3])
    s_z = tf.random_uniform([],minval = scale_range[4],maxval = scale_range[5])    
    #rotation param
    yam = tf.random_uniform([],minval = degree_range[0],maxval = degree_range[1])
    pitch = tf.random_uniform([],minval = degree_range[2],maxval = degree_range[3])
    roll = tf.random_uniform([],minval = degree_range[4],maxval = degree_range[5])
    #translation param
    t_x = tf.random_uniform([],minval = translaton_range[0],maxval = translaton_range[1])
    t_y = tf.random_uniform([],minval = translaton_range[2],maxval = translaton_range[3])
    t_z = tf.random_uniform([],minval = translaton_range[4],maxval = translaton_range[5])

    I = tf.convert_to_tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    R_z = tf.convert_to_tensor([[[tf.cos(yam), -tf.sin(yam), 0.0], [tf.sin(yam), tf.cos(yam), 0.0], [0.0, 0.0, 1.0]]])
    R_y = tf.convert_to_tensor([[[tf.cos(pitch), 0.0, tf.sin(pitch)], [0.0, 1.0, 0.0], [-tf.sin(pitch), 0.0, tf.cos(pitch)]]])
    R_x = tf.convert_to_tensor([[[1.0, 0.0, 0.0], [0.0, tf.cos(roll), -tf.sin(roll)], [0.0, tf.sin(roll), tf.cos(roll)]]])

    S   = tf.convert_to_tensor([[[s_x, 0.0, 0.0], [0.0, s_y, 0.0], [0.0, 0.0, s_z]]])
    b = tf.convert_to_tensor([t_x,t_y,t_z])
    b = tf.reshape(b, [-1, 3])
    A = tf.matmul(R_z,R_y)
    A = tf.matmul(A,R_x)
    A = tf.matmul(A,S)
    # the flow is displacement(x) = place(x) - x = (Ax + b) - x
    W = A - I
    if sx == None:
        sx, sy, sz = img.shape.as_list()[1:4]
    flow = affine_flow(W, b, sx, sy, sz)
    #reconstruction = Dense3DSpatialTransformer()
    #transformed_img = reconstruction([img, flow])
    return flow

def random_intensity(img):
    beta = tf.random_uniform([],minval = 0.5,maxval = 1.5)
    #mean, var = tf.nn.moments(img,axes=[1,2,3,4])
    Max = tf.reduce_max(img)
    img = img/Max
    img = tf.pow(img ,beta)
    img = img*Max
    #mean, var = tf.nn.moments(img,axes=[1,2,3,4])
    #img = (img - mean)/var
    return img

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

def affine_intensity(img,sx=None, sy=None, sz=None):
    affine_img,flow = random_affine(img,sx=sx, sy=sy, sz=sy)
    result = random_intensity(affine_img)
    return result, flow


def extract_ampl_phase(img):
    ft_amp = tf.abs(img)
    fft_phase = tf.angle(img)
    return ft_amp, fft_phase

def low_freq_mutate(amp_src, amp_trg, L=0.1):
    shape = [1]+amp_trg.get_shape().as_list()[1:]
    a_src = fftshift(amp_src,[2,3,4])
    a_trg = fftshift(amp_trg,[2,3,4])
    mask = tf.Variable(tf.ones(shape, dtype = tf.complex64))
    b = tf.cast((min(shape[2:5])*L),tf.int32)
    c_x = shape[2]//2
    c_y = shape[3]//2
    c_z = shape[4]//2

    x1 = c_x - b + 1
    x2 = c_x + b
    y1 = c_y - b
    y2 = c_y + b + 1
    z1 = c_z - b
    z2 = c_z + b + 1
    print(b)
    
    mask1 = mask[:,:,x1:x2, y1:y2, z1:z2].assign(tf.zeros_like(mask, dtype = tf.complex64)[:,:,x1:x2, y1:y2, z1:z2])
    mask1 = mask1+tf.assign(mask,tf.ones(shape, dtype = tf.complex64))-1
    a_src = mask1*a_src + (1-mask1)*a_trg

    a_src_ = fftshift(a_src,[2,3,4])
    return a_src_, mask1

def FDA_S2T2(src_img, trg_img, L = 0.0, if_random=False):
    src_img = tf.cast(tf.transpose(src_img, (0,4,1,2,3)), tf.complex64)
    trg_img = tf.cast(tf.transpose(trg_img, (0,4,1,2,3)), tf.complex64)
    fft_src = tf.signal.fft3d(src_img)
    fft_trg = tf.signal.fft3d(trg_img)
    #extract amplitude and phase
    amp_src, pha_src = extract_ampl_phase( fft_src)
    amp_trg, pha_trg = extract_ampl_phase( fft_trg)
    if if_random:
      L = tf.random_uniform([],minval = 0.0,maxval = 1.0)
    amp_src_ = (1-L)*amp_src + L*amp_trg
    fft_src_real = amp_src_ * tf.cos(pha_src)
    fft_src_imag = amp_src_ * tf.sin(pha_src)
    fft_src_ = tf.complex(fft_src_real, fft_src_imag)
    src_in_trg = tf.signal.ifft3d(fft_src_)
    src_in_trg = tf.cast(tf.transpose(src_in_trg, (0,2,3,4,1)), tf.float32)
    src_in_trg = tf.where(src_in_trg>0,src_in_trg,tf.zeros_like(src_in_trg))
    src_in_trg = tf.where(src_in_trg<1,src_in_trg,tf.zeros_like(src_in_trg))
    return src_in_trg

def FDA_S2T(src_img, trg_img, L = 1.0, if_random=False):
    src_img = tf.cast(tf.transpose(src_img, (0,4,1,2,3)), tf.complex64)
    trg_img = tf.cast(tf.transpose(trg_img, (0,4,1,2,3)), tf.complex64)
    fft_src = tf.signal.fft3d(src_img)
    fft_trg = tf.signal.fft3d(trg_img)
    if if_random:
      L = tf.random_uniform([],minval = 0.05,maxval = 0.2)
    fft_src_, a_src= low_freq_mutate(fft_src, fft_trg, L=L)

    src_in_trg = tf.signal.ifft3d(fft_src_)
    src_in_trg = tf.cast(tf.transpose(src_in_trg, (0,2,3,4,1)), tf.float32)
    src_in_trg = tf.where(src_in_trg>0,src_in_trg,tf.zeros_like(src_in_trg))
    src_in_trg = tf.where(src_in_trg<1,src_in_trg,tf.zeros_like(src_in_trg))
    return src_in_trg


def fftshift(x, axes=None, name=None):
  with _ops.name_scope(name, "fftshift") as name:
    x = _ops.convert_to_tensor(x)
    if axes is None:
      axes = tuple(range(x.shape.ndims))
      shift = [int(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
      shift = int(x.shape[axes] // 2)
    else:
      #shift = [int((x.shape[ax]) // 2) for ax in axes]
      shift = [64,64,64]

    return manip_ops.roll(x, shift, axes)
