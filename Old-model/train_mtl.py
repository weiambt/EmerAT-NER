import time
import numpy as np
import os
import tensorflow_addons as tfa

import tensorflow as tfv2
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()

import mtl_model
from common import CommonUtil


def main(_):
    # 如果不存在,创建文件
    dir_path = './ckpt/ner-cws-2025-01-03'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # model_name = 'model-{}'.format(dir_path)
    out = open('{}/log.txt'.format(dir_path), 'w')
    save_path = '{}/model'.format(dir_path)
    startTime = time.time()

    print ('read word embedding......')
    embedding=np.load('./data/weibo_vector.npy')
    print ('read ner train data......')
    train_word=np.load('./data/weibo_train_word.npy')
    train_label=np.load('./data/weibo_train_label.npy')
    train_length=np.load('./data/weibo_train_length.npy')

    print ('read ner test data......')
    test_word = np.load('./data/weibo_test_word.npy')
    test_label = np.load('./data/weibo_test_label.npy')
    test_length = np.load('./data/weibo_test_length.npy')

    print ('read cws train data......')
    train_cws_word=np.load('./data/weibo_cws_word.npy')
    train_cws_label=np.load('./data/weibo_cws_label.npy')
    train_cws_length=np.load('./data/weibo_cws_length.npy')
    setting = mtl_model.Setting()
    task_ner=[]
    task_cws=[]
    for i in range(setting.batch_size):
        task_ner.append([1,0])
        task_cws.append([0,1])

    with tf.Graph().as_default():
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess=tf.Session(config=config)
        with sess.as_default():
            initializer = tfv2.keras.initializers.GlorotUniform()
            with tf.variable_scope('ner_model',reuse=None,initializer=initializer):
                model = mtl_model.TransferModel(setting, tf.cast(embedding, tf.float32), adv=True, is_train=True)
                model.multi_task()
            global_step = tf.Variable(0, name="global_step", trainable=False)
            global_step1 = tf.Variable(0, name="global_step1", trainable=False)
            optimizer = tf.train.AdamOptimizer(0.001)
            if setting.clip>0:
                grads, vs=zip(*optimizer.compute_gradients(model.loss))
                grads,_ =tf.clip_by_global_norm(grads,clip_norm=setting.clip)
                train_op=optimizer.apply_gradients(zip(grads,vs),global_step)
                train_op1 = optimizer.apply_gradients(zip(grads, vs), global_step1)
            else:
                train_op=optimizer.minimize(model.loss, global_step)
                train_op1 = optimizer.minimize(model.loss, global_step1)

            sess.run(tf.initialize_all_variables())
            saver=tf.train.Saver(max_to_keep=3)
            best_f1_val = 0.0
            best_step = -1

            # each epoch
            for one_epoch in range(setting.num_epoches):
                info = "------epoch {}\n".format(one_epoch)
                print(info)
                out.write(info)
                temp_order=list(range(len(train_word)))
                temp_order_cws=list(range(len(train_cws_word)))
                np.random.shuffle(temp_order)
                np.random.shuffle(temp_order_cws)
                for i in range(len(temp_order)//setting.batch_size):
                    for j in range(2):
                        # if j==0:
                        #     continue
                        if j==0:
                            temp_word = []
                            temp_label = []
                            temp_label_=[]
                            temp_length = []
                            temp_input_index = temp_order[i * setting.batch_size:(i + 1) * setting.batch_size]
                            temp_input_index1 = temp_order_cws[i * setting.batch_size:(i + 1) * setting.batch_size]
                            for k in range(len(temp_input_index)):
                                temp_word.append(train_word[temp_input_index[k]])
                                temp_label.append(train_label[temp_input_index[k]])
                                temp_label_.append(train_cws_label[temp_input_index1[k]])
                                temp_length.append(train_length[temp_input_index[k]])
                            feed_dict={}
                            feed_dict[model.input]=np.asarray(temp_word)
                            feed_dict[model.label]=np.asarray(temp_label)
                            # 没必要传
                            feed_dict[model.label_]=np.asarray(temp_label_)
                            feed_dict[model.sent_len]=np.asarray(temp_length)
                            feed_dict[model.is_ner]=1
                            feed_dict[model.task_label]=np.asarray(task_ner)
                            _, step, loss= sess.run([train_op,global_step,model.ner_loss],feed_dict)
                            if step % 20 ==0:
                                temp = "step {},loss {}\n".format(step, loss)
                                print (temp)
                                out.write(temp)
                            current_step = step
                            # if current_step % 120 == 0 and current_step > 2000 and current_step < 10000:
                            if current_step % 40 == 0:
                                saver.save(sess, save_path=save_path, global_step=current_step)
                        else:
                            temp_cws_word = []
                            temp_cws_label = []
                            temp_cws_label_=[]
                            temp_cws_length = []
                            temp_input_index = temp_order_cws[i * setting.batch_size:(i + 1) * setting.batch_size]
                            temp_input_index1 = temp_order[i * setting.batch_size:(i + 1) * setting.batch_size]
                            for k in range(len(temp_input_index)):
                                temp_cws_word.append(train_cws_word[temp_input_index[k]])
                                temp_cws_label.append(train_label[temp_input_index1[k]])
                                temp_cws_label_.append(train_cws_label[temp_input_index[k]])
                                temp_cws_length.append(train_cws_length[temp_input_index[k]])
                            feed_dict = {}
                            feed_dict[model.input] = np.asarray(temp_cws_word)
                            feed_dict[model.label] = np.asarray(temp_cws_label)
                            feed_dict[model.label_]=np.asarray(temp_cws_label_)
                            feed_dict[model.sent_len] = np.asarray(temp_cws_length)
                            feed_dict[model.is_ner] = 0
                            feed_dict[model.task_label] = np.asarray(task_cws)
                            _, step1, cws_loss= sess.run([train_op1,global_step1,model.cws_loss], feed_dict)
                            # if step1 % 500 ==0:
                            if step1 % 20 ==0:
                                temp = "step2 {},cws_loss {}\n\n".format(step1, cws_loss)
                                print (temp)

                # 每个epoch后验证模型
                f1_val = validate(sess, test_word, test_label, test_length, setting, model)
                if f1_val > best_f1_val:
                    best_f1_val = f1_val
                    best_step = current_step
                    info = "=====update best F1, epoch = {},step = {} ,best_f1_score = {},current f1_val > best_f1_score".format(
                        one_epoch, step, best_f1_val)
                    print(info)
                    out.write(info + "\n")
                    saver.save(sess, save_path=save_path, global_step=current_step)
            info = '========= final best f1 = {},on epoch {},on step {}'.format(best_f1_val, one_epoch,best_step)
            print(info)
            out.write(info + "\n")

    extime = time.time() - startTime
    info = 'execute time is {} s\n,'.format(str(int(extime)))
    print(info)
    out.write(info)
    out.close()


def validate(sess,test_word,test_label,test_length,setting,model):
    results = []
    for j in range(len(test_word) // setting.batch_size):
        word_batch = test_word[j * setting.batch_size:(j + 1) * setting.batch_size]
        length_batch = test_length[j * setting.batch_size:(j + 1) * setting.batch_size]
        label_batch = test_label[j * setting.batch_size:(j + 1) * setting.batch_size]
        feed_dict = {}
        feed_dict[model.input] = word_batch
        feed_dict[model.sent_len] = length_batch
        feed_dict[model.is_ner] = 1
        logits, trans_params = sess.run([model.ner_project_logits, model.ner_trans_params], feed_dict)
        # viterbi_sequences=decode(logits,trans_params,length_batch)
        viterbi_sequences, viterbi_scores = tfa.text.crf_decode(logits, trans_params, length_batch)

        result_batch = CommonUtil.evaluate(viterbi_sequences, label_batch, length_batch, word_batch)
        results.append(result_batch)
    f1 = CommonUtil.compute_f1(results)
    return f1

if __name__ == "__main__":
    tf.app.run()