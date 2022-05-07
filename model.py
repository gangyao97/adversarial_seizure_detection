# -*- coding: utf-8 -*-

import tensorflow as tf

def init_tf_placeholder(params,no_fea_long):
    '''
    Parameters
    ----------
    params : Model Hypermeters.
    no_fea_long: No. fea_long

    Description:
    ------------
    To initial the model placeholders

    Returns
    -------
    xs  :    Input placeholder
    ys_t :   task output placeholder
    ys_p :   person output placeholder
    keep_prob : drop-out placeholder
    '''
    tf.compat.v1.disable_eager_execution()
    xs = tf.compat.v1.placeholder(tf.float32, [None, no_fea_long], name='xsss')  # 249*100
    ys_t = tf.compat.v1.placeholder(tf.float32, [None, params["n_class_t"]], name='ys_t')
    ys_p = tf.compat.v1.placeholder(tf.float32, [None, params["n_person_"]], name='ys_p')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep')
    return xs,ys_t,ys_p,keep_prob

def CNN_AE(params,AE,reuse,xs,keep_prob):
    with tf.compat.v1.variable_scope("CNN_AE", reuse=reuse):
        # dim_code = 1000
        seg_length = params["seg_length"]
        no_fea = params["no_fea"]
        input = tf.reshape(xs, [-1, params["seg_length"], params["no_fea"], 1])  # [200, 14]
        input = tf.contrib.layers.batch_norm(input, decay=0.9)
        #input = tf.compat.v1.layers.batch_normalization(input, decay=0.9)
        input = tf.nn.dropout(input, keep_prob)

        print(xs.shape)  # [n_samples, 28,28,1]

        depth_AE = params["depth_AE"]  # default is 8
        conv1 = tf.layers.conv2d(inputs=input, filters=depth_AE, kernel_size=params["AE_Kernel_size"], padding="same",
                                 activation=tf.nn.relu)
        h_t = tf.layers.max_pooling2d(inputs=conv1, pool_size=params["pool_siz"], strides=params["strides"])
        # pool1 = tf.contrib.layers.batch_norm(pool1, decay=0.9)

        conv1_p = tf.layers.conv2d(inputs=input, filters=depth_AE, kernel_size=params["AE_Kernel_size"], padding="same",
                                   activation=tf.nn.relu)
        h_p = tf.layers.max_pooling2d(inputs=conv1_p, pool_size=params["pool_siz"], strides=params["strides"])

        # decoder
        output_t = tf.layers.conv2d_transpose(h_t, kernel_size=params["decoder_kernel"], filters=params["decoder_filter"], strides=params["decoder_strider"], padding='same')
        # output_t = tf.nn.relu(tf.contrib.layers.batch_norm(output_t, decay=0.9))
        output_p = tf.layers.conv2d_transpose(h_p, kernel_size=params["decoder_kernel"], filters=params["decoder_filter"], strides=params["decoder_strider"], padding='same')
        # output_p = tf.nn.relu(tf.contrib.layers.batch_norm(output_p, decay=0.9))
        output = (output_t + output_p) / 2

        # #another decoder
        # h = (h_t + h_p)/2
        # output = tf.layers.conv2d_transpose(h, kernel_size=5, filters=1, strides=[l_AE, w_AE], padding='same')
        # output = tf.nn.relu(tf.contrib.layers.batch_norm(output, decay=0.9))

        output = tf.reshape(output, [-1, seg_length * no_fea])
        return h_t,h_p,output

def CNN_Class_p(params,h_p,keep_prob,reuse):

    with tf.compat.v1.variable_scope("CNN_Class_p", reuse=reuse):
        n_class_p = params["n_person_"]
        
        x_image = tf.contrib.layers.batch_norm(h_p, decay=0.9)
    
            # x_image_t = tf.nn.dropout(x_image_t, keep_prob)
        print(x_image.shape)  # [n_samples, 28,28,1]
        depth_1 = params["depth_1"]  # default is 8
        conv1 = tf.layers.conv2d(inputs=x_image, filters=depth_1, kernel_size=params["depth1_2_kernel"], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=params["pool1_size"], strides=params["pool1_strides"])
        pool1 = tf.contrib.layers.batch_norm(pool1, decay=0.9)
    
        depth_2 = params["depth_2"]  # default is 32
        conv2 = tf.layers.conv2d(inputs=pool1, filters=depth_2, kernel_size=params["depth1_2_kernel"], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=params["pool2_size"], strides=params["pool2_strides"])
        pool2 = tf.contrib.layers.batch_norm(pool2, decay=0.9)
    
        depth_3 = params["depth_3"]
        conv3 = tf.layers.conv2d(inputs=pool2, filters=depth_3, kernel_size=params["depth3_4_kernel"], padding="same", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=params["pool3_size"], strides=params["pool3_strides"])
        pool3 = tf.contrib.layers.batch_norm(pool3, decay=0.9)
        # print(pool1.get_shape(), pool2.get_shape(), pool3.get_shape(),)
    
        depth_4 = params["depth_4"]
        conv4 = tf.layers.conv2d(inputs=pool3, filters=depth_4, kernel_size=params["depth3_4_kernel"], padding="same", activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=params["pool4_size"], strides=params["pool4_strides"])
        pool4 = tf.contrib.layers.batch_norm(pool4, decay=0.9)
    
        fc1 = tf.contrib.layers.flatten(pool4)  # flatten the pool 2
        print(pool1.get_shape(), pool2.get_shape(), pool3.get_shape(),  pool4.get_shape(), fc1.get_shape())
    
    
        # """Add another FC layer"""
        dim_hidden1 = params["fc1_hidden"]
        fc1 = tf.layers.dense(fc1, units=dim_hidden1, activation=tf.nn.sigmoid)
        fc1 = tf.nn.dropout(fc1, keep_prob)
    
        dim_hidden3 = params["fc3_hidden"]
        fc3 = tf.layers.dense(fc1, units=dim_hidden3, activation=tf.nn.sigmoid)
        fc3 = tf.nn.dropout(fc3, keep_prob)
        prediction_p = tf.layers.dense(fc3, units=n_class_p, activation=None)
        print('prediction_p', tf.shape(prediction_p))
        return prediction_p

def CNN_Class_t(params,h_t,xs,keep_prob,reuse):
    with tf.compat.v1.variable_scope("CNN_Class_t", reuse=reuse):
        n_class_p = params["n_person_"]
        
        x_image = tf.contrib.layers.batch_norm(h_t, decay=0.9)
    
            # x_image_t = tf.nn.dropout(x_image_t, keep_prob)
        print(x_image.shape)  # [n_samples, 28,28,1]
        depth_1 = params["depth_1"]  # default is 8
        conv1 = tf.layers.conv2d(inputs=x_image, filters=depth_1, kernel_size=params["depth1_2_kernel"], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=params["pool1_size"], strides=params["pool1_strides"])
        pool1 = tf.contrib.layers.batch_norm(pool1, decay=0.9)
    
        depth_2 = params["depth_2"]  # default is 32
        conv2 = tf.layers.conv2d(inputs=pool1, filters=depth_2, kernel_size=params["depth1_2_kernel"], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=params["pool2_size"], strides=params["pool2_strides"])
        pool2 = tf.contrib.layers.batch_norm(pool2, decay=0.9)
    
        depth_3 = params["depth_3"]
        conv3 = tf.layers.conv2d(inputs=pool2, filters=depth_3, kernel_size=params["depth3_4_kernel"], padding="same", activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=params["pool3_size"], strides=params["pool3_strides"])
        pool3 = tf.contrib.layers.batch_norm(pool3, decay=0.9)
        # print(pool1.get_shape(), pool2.get_shape(), pool3.get_shape(),)
    
        depth_4 = params["depth_4"]
        conv4 = tf.layers.conv2d(inputs=pool3, filters=depth_4, kernel_size=params["depth3_4_kernel"], padding="same", activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=params["pool4_size"], strides=params["pool4_strides"])
        pool4 = tf.contrib.layers.batch_norm(pool4, decay=0.9)
    
        fc1 = tf.contrib.layers.flatten(pool4)  # flatten the pool 2
        print(pool1.get_shape(), pool2.get_shape(), pool3.get_shape(),  pool4.get_shape(), fc1.get_shape())
    
    
        # """Add another FC layer"""
        dim_hidden1 = params["fc1_hidden"]
        fc1 = tf.layers.dense(fc1, units=dim_hidden1, activation=tf.nn.sigmoid)
        fc1 = tf.nn.dropout(fc1, keep_prob)
    
        dim_hidden3 = params["fc3_hidden"]
        fc3 = tf.layers.dense(fc1, units=dim_hidden3, activation=tf.nn.sigmoid)
        fc3 = tf.nn.dropout(fc3, keep_prob)
        
        # Attention layer
        att = tf.layers.dense(xs, units=fc3.shape[-1], activation=tf.nn.sigmoid)
        fc3 = tf.multiply(fc3, att)
        n_class_t = params["n_class_t"]
        prediction_t = tf.layers.dense(fc3, units=n_class_t, activation=None)
        print('prediction_t', prediction_t.get_shape())
        return att, prediction_t