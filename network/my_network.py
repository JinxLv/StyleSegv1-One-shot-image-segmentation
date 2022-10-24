import tensorflow as tf
import tflearn
import numpy as np

def CosineDisMap(I,J):
    amb = lambda x:tf.pow(tf.reduce_sum(x*x,axis=-1, keep_dims=True),0.5)
    distmap = 1 - tf.abs(tf.reduce_sum(I*J,axis=-1, keep_dims=True))/(amb(I)*amb(J))
    return distmap
def get_bias(input_f, input_p, bias_x,bias_y,bias_z, r=3):
    d = 2*r+1
    '''f_size = (2*abs(bias_x)+1, 2*abs(bias_y)+1, 2*abs(bias_z)+1,1,1)
    f_n = np.zeros(f_size)
    f_n[bias_x+abs(bias_x),bias_y+abs(bias_y),bias_z+abs(bias_z)] = 1
    f_n = tf.convert_to_tensor(f_n, dtype=tf.float32)

    f_n = [[[0]*d]*d]*d
    f_n[bias_x+r][bias_y+r][bias_z+r] = 1
    f_n = tf.convert_to_tensor(f_n, dtype=tf.float32)
    f_n = tf.expand_dims(tf.expand_dims(f_n,-1),-1)
    '''
    x,y,z = 2*abs(bias_x)+1, 2*abs(bias_y)+1, 2*abs(bias_z)+1
    f_n=[[[0] * z for i in range(y)] for i in range(x)]
    f_n[bias_x+x//2][bias_y+y//2][bias_z+z//2] = 1
    f_n = tf.convert_to_tensor(f_n, dtype=tf.float32)
    f_n = tf.expand_dims(tf.expand_dims(f_n,-1),-1)
    input_f = tf.transpose(input_f, [4,1,2,3,0])
    output_f = tf.nn.conv3d(input_f, f_n, strides=[1, 1, 1, 1, 1], padding='SAME')
    output_f = tf.transpose(output_f, [4,1,2,3,0])
    input_p = tf.transpose(input_p, [4,1,2,3,0])
    output_p = tf.nn.conv3d(input_p, f_n, strides=[1, 1, 1, 1, 1], padding='SAME')
    output_p = tf.transpose(output_p, [4,1,2,3,0])
    return output_f, output_p
def Correlation_Module(F_q, F_s, P_s,r=3):
    batch, W, H, D, N = F_q.get_shape().as_list()
    P_confident = tf.cond(tflearn.get_training_mode(), lambda: P_s*0.0, lambda: P_s)
    P_counter = tf.cond(tflearn.get_training_mode(), lambda: P_s*0.0, lambda: P_s)
    corr_map = []
    counter = 0
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            for k in range(-r, r+1):
                if i==0 and j==0 and k==0:
                    continue
                bias_F_s, bias_P_s = get_bias(F_s, P_s, i, j, k, r)
                counter += 1
                P_counter += bias_P_s
                #corr = tf.sigmoid(tf.reduce_mean(bias_F_s*F_q,axis=-1,keep_dims=True))
                corr = 1 - CosineDisMap(bias_F_s, F_q)
                corr_map.append(corr)
                P_confident += bias_P_s*corr
    P_confident /= P_counter
    corr_map = tf.concat(corr_map,axis=-1)
    correlation_feature = tf.concat([corr_map, P_confident],axis=-1)
    return P_confident,corr
    

    
    


