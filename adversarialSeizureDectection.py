from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import pickle
from sklearn import preprocessing
import pandas as pd
from pathlib import Path
from datetime import datetime
from utils import *
from model import *
import logging

params_file = 'params.json'
params = load_param(params_file)
def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


def extract(input, n_fea, time_window, moving):
    global n_classes
    xx = input[:, :n_fea]
    yy = input[:, n_fea:n_fea + 1]
    new_x = []
    new_y = []
    number = int((xx.shape[0] / moving) - 1)
    for i in range(number):
        ave_y = np.average(yy[int(i * moving):int(i * moving + time_window)])
        if ave_y in range(n_classes + 1):
            new_x.append(xx[int(i * moving):int(i * moving + time_window), :])
            new_y.append(ave_y)
        else:
            new_x.append(xx[int(i * moving):int(i * moving + time_window), :])
            new_y.append(0)

    new_x = np.array(new_x)
    new_x = new_x.reshape([-1, n_fea * time_window])
    new_y = np.array(new_y)
    new_y.shape = [new_y.shape[0], 1]
    data = np.hstack((new_x, new_y))
    data = np.vstack((data, data[-1]))  # add the last sample again, to make the sample number round
    return data

def compute_accuracy_t(v_xs, v_ys):  # this function only calculate the acc of CNN_task
    global prediction_t
    y_pre = sess.run(prediction_t, feed_dict={xs: v_xs, keep_prob: keep})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys_t: v_ys, keep_prob: keep})
    return result


def compute_accuracy_p(v_xs, v_ys):  # this function only calculate the acc of CNN_task
    global prediction_p
    y_pre = sess.run(prediction_p, feed_dict={xs: v_xs, keep_prob: keep})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys_p: v_ys, keep_prob: keep})
    return result

date = datetime.now().strftime("%Y_%m_%d-%I-%M_%p")
filename = f"test_{date}"+'.csv'
logger_name = f"train_{date}"+'.log'
print(params["AE_Kernel_size"],params["sample_persub"],params["padding"],params["conv_activation"])
headerList = ['P_ID','Step','train_acc_task','train_acc_person','test_acc_task','testing_AE','testing_t',
                          'training_cost_total','training_cost_AE','training_cost_t','training_cost_p']
logger=set_logger(logger_name)
log_hyperparameter(logger,params_file)
# # WoW! use this to limit the GPU number
# import os
# GPU_ID = 2
# os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
# print('Let`s start!, GPU:', GPU_ID)

################THU seizure data reading
# thr first 21 columns are features, the 22nd column is serzure/normal, the 23rd column is person label.
# in the task lable, 0: normal, 1: seizure

# data reading
# python 3: add ',encoding='iso-8859-1'' in the pickle.load.
all_data = pickle.load(open("all_14sub.p", "rb" ),encoding = 'iso-8859-1' )
print(type(all_data), all_data.shape, all_data[:, -1])

n_classes = 2
n_person_ = 13  # the number of training subjects
sample_persub = 250*500  # we have overlapping now
print(type(all_data), all_data.shape, all_data[:, -1])

no_fea = 21  # data.shape[-1] - 1
seg_length = 250  # # 255 for raw data, 96 for layer 23, 64 for layer 2, 32 for layer 2

scaler = preprocessing.MinMaxScaler()  # normalization
F = scaler.fit_transform(all_data[:, :no_fea])  # scale to [0, 1]

all_data = np.hstack((F, all_data[:, no_fea:no_fea+1]))  # only use the task ID


"""Make person label"""
n_sample_ = int(2*sample_persub/seg_length )  # the number of sampls of each subject after reshape
ll = np.ones([n_sample_, 1])*0
for hh in range(1, n_person_):
    ll_new = np.ones([n_sample_, 1])*hh
    ll = np.vstack((ll, ll_new))
print('the shape of maked person label', ll.shape)

ll_test = np.ones([n_sample_, 1])*n_person_

ss_train = time.time()
n_class_t = params["n_class_t"]  # 0-3
n_class_p = params["n_person_"]  # 0-8
keep = params["Drop_out_keep"]

# Person Independent
for P_ID in range(14):  # n_person_++1
    if P_ID==0:
        reuse=False
    else:
        reuse=True
    """Select train and test subject"""
    
    
    data_ = all_data[sample_persub*P_ID:sample_persub*(P_ID+1)]

    list = range(sample_persub*P_ID, sample_persub*(P_ID+1))
    data = np.delete(all_data, list, axis=0)
    # overlap
    train_data = extract(data, n_fea=no_fea, time_window=seg_length, moving=(seg_length/2))
    test_data = extract(data_, n_fea=no_fea, time_window=seg_length, moving=(seg_length/2))  # 50% overlapping
    # continue
    """Replace the original person data by the maked data"""
    no_fea_long = train_data.shape[-1] - 1  # here is - 2, because has two IDs
    print("no_fea_long==",no_fea_long)
    print(train_data[:, :no_fea_long+1].shape, ll.shape)
    train_data = np.hstack((train_data[:, :no_fea_long+1], ll))
    test_data = np.hstack((test_data[:, :no_fea_long + 1], ll_test))
    print(train_data.shape)
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    print(train_data.shape, test_data.shape)

    feature_train = train_data[:, :no_fea_long]
    feature_test = test_data[:, :no_fea_long]
    label_train_t = train_data[:, no_fea_long:no_fea_long + 1]
    label_test_t = test_data[:, no_fea_long:no_fea_long + 1]
    label_train_p = train_data[:, no_fea_long + 1:no_fea_long + 2]

    label_train_t = one_hot(label_train_t)
    label_test_t = one_hot(label_test_t)
    label_train_p = one_hot(label_train_p)



    a = feature_train

    ## batch split
    batch_size = int(feature_test.shape[0])
    train_fea = []
    n_group = int(feature_train.shape[0]/feature_test.shape[0])
    for i in range(n_group):
        f = a[(0+batch_size*i):(batch_size+batch_size*i)]
        train_fea.append(f)
    print (train_fea[0].shape)

    train_label_t=[]
    for i in range(n_group):
        f = label_train_t[(0 + batch_size * i):(batch_size + batch_size * i), :]
        train_label_t.append(f)
    print (train_label_t[0].shape)

    train_label_p = []
    for i in range(n_group):
        f = label_train_p[(0 + batch_size * i):(batch_size + batch_size * i), :]
        train_label_p.append(f)
    print (train_label_p[0].shape)
    

    """Placeholder"""
    # define placeholder for inputs to network
   
    xs,ys_t,ys_p,keep_prob = init_tf_placeholder(params,no_fea_long)
    #CNN code for AE
    h_t,h_p,output = CNN_AE(params,"AE",reuse,xs,keep_prob)   

    """CNN code for task, maybe we can make it deeper? """
    
    att, prediction_t = CNN_Class_t(params,h_t,xs,keep_prob,reuse)
    """CNN code for person"""
    
    prediction_p  = CNN_Class_p(params,h_p,keep_prob,reuse)
    def kl_divergence(p, q):
        return tf.reduce_sum(p * tf.log(p/q))


    """cost calculation"""
    train_vars = tf.compat.v1.trainable_variables() 
    l2_coefficient = params["l2_coefficient"]
    l2_AE = l2_coefficient * sum(tf.nn.l2_loss(var) for var in train_vars if var.name.startswith("CNN_AE")) #0.005
    l2_class = l2_coefficient * sum(tf.nn.l2_loss(var) for var in train_vars if var.name.startswith("CNN_Class_p"))
    """multiply 10 to enhance the cross_entropy_t """
    cross_entropy_t = 10*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_t, labels=ys_t)) 
    cross_entropy_p = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_p, labels=ys_p))
 
    cost_AE = (tf.reduce_mean(tf.pow(xs - output, 2)) + l2_AE)

    class_vars = [var for var in train_vars if var.name.startswith("CNN_Class_p")]  
    AE_vars = [var for var in train_vars if var.name.startswith("CNN_AE")]
    t_vars = [var for var in train_vars if var.name.startswith("CNN_Class_t")]

    cost = cost_AE  + cross_entropy_t +   cross_entropy_p + l2_class + l2_AE
    lr = params["learning_rate"]
    
    with tf.compat.v1.variable_scope("optimization", reuse=reuse):
        train_step_task = tf.compat.v1.train.AdamOptimizer(lr).minimize(cost) 
      
        train_step_t =  tf.compat.v1.train.AdamOptimizer(lr).minimize(cross_entropy_t) 
    
    
    con = tf.compat.v1.ConfigProto() 
    con.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=con) 
    '''
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    '''
    
    init = tf.compat.v1.global_variables_initializer() #tf.global_variables_initializer()
    sess.run(init)
    # History records of loss functions
    cost_his = []
    cost_AE_his = []
    cost_t_his = []
    cost_p_his = []

    test_cost_t_his = []

    start=time.time()
    step = 1
    num_epochs = params["num_epochs"]
    while step < num_epochs:  # 250 iterations
        print('iteration step', step)
        for i in range(n_group):
            feed = {xs: train_fea[i], ys_t: train_label_t[i], ys_p: train_label_p[i], keep_prob:keep}
            sess.run(train_step_task, feed_dict=feed)
            sess.run(train_step_t, feed_dict=feed)
        print("iteration Running Time,", time.time()-start)
        if step % 5 == 0:
            """training cost"""
            cost_, cost_AE_, cross_entropy_p_, cross_entropy_t_=sess.run([cost, cost_AE, cross_entropy_p, cross_entropy_t],
                                                                         feed_dict={xs: train_fea[0],
                                             ys_t: train_label_t[0], ys_p: train_label_p[0], keep_prob:keep})

            """testing cost"""
            cost_AE_test_, cross_entropy_t_test_=sess.run([cost_AE, cross_entropy_t],
                                                          feed_dict ={xs: feature_test, ys_t: label_test_t, keep_prob: keep})

            train_acc_task = compute_accuracy_t(feature_train, label_train_t)
            train_acc_person = compute_accuracy_p(feature_train, label_train_p)
            test_acc_task = compute_accuracy_t(feature_test, label_test_t)
            print('person, step:',P_ID, step, 'train acc task', train_acc_task ,
                  'train acc person', train_acc_person,
                  ',the test acc task', test_acc_task ,
                  'testing: AE, t', cost_AE_test_, cross_entropy_t_test_)

            print('training cost: total, AE, t, p',cost_, cost_AE_,  cross_entropy_t_, cross_entropy_p_)
            
             
            train_results = []
            train_results.append(P_ID)
            train_results.extend([step,train_acc_task,train_acc_person,test_acc_task,cost_AE_test_,
                                 cross_entropy_t_test_,cost_, cost_AE_,cross_entropy_t_,cross_entropy_p_])
            save_data(filename, headerList,train_results)
            
            cost_his.append(cost_)
            cost_AE_his.append(cost_AE_)
            
            cost_t_his.append(cross_entropy_t_)
            cost_p_his.append(cross_entropy_p_)

            test_cost_t_his.append(cross_entropy_t_test_)

        # save the attention weights for fine-grained analysis
        if step % 50 == 0:
            att_ = sess.run(att, feed_dict={xs: feature_test, ys_t: label_test_t, keep_prob: keep})

            ss = time.time()
            pred = sess.run(prediction_t, feed_dict={xs: feature_test, ys_t: label_test_t, keep_prob: keep})
            print('training, testing time', time.time()-ss_train, time.time()-ss)

            pickle.dump(att_,  open('TUH_attention_P'
                                    +str(step)+date+'_backup.p',  "wb"), protocol=2)
            print('attention saved, person:', P_ID)

        step += 1
    print("PID ",P_ID,"Training Time", time.time()-start)
    # save the cost history values for convergence analysis
    pickle.dump(cost_his, open('cost_his.p', "wb"))
    pickle.dump(cost_AE_his,
                open('cost_AE_his.p', "wb"))
    pickle.dump(cost_t_his,
                open('cost_t_his.p', "wb"))
    pickle.dump(cost_p_his,
                open('cost_p_his.p', "wb"))
    pickle.dump(test_cost_t_his,
                open('test_cost_t_his.p', "wb"))
    print("five losses are saved at /home/xiangzhang/scratch/activity_recognition_practice/Parkinson_seizure")


print("total training time",time.time()-ss_train)
logger.info("total training time-- %s",time.time()-ss_train)
logging.shutdown()
