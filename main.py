import argparse
import time
import tensorflow as tf
import numpy as np
import pickle

from data_utils import *
from model import PACA




def forward_evaluate(model,data,iterator):
    recall = np.zeros(4)
    mrr = np.zeros(4)
    evalutation_point_count = 0
    ranks=[]
    length_bat=[]
    for _, test_index in iterator:
        x, mask, lengths, y = prepare_data([data[0][t] for t in test_index],
                                           np.array(data[1])[test_index])
        dropouts = [0,0,0]
        outs = sess.run([model.recall, model.mrr,model.ranks],
                         feed_dict={model.x: x, model.mask: mask, model.labels: y
                             ,  model.dropouts:dropouts})
        for i in range(4):
            recall[i] += outs[0][i]
            mrr[i] += outs[1][i]
        evalutation_point_count += x.shape[1]
        ranks.extend(outs[2].tolist())
        length_bat.extend(lengths)
    with open("./test_rank.txt","wb") as f:
        pickle.dump((ranks,length_bat),f)
    recall = recall / evalutation_point_count
    mrr = mrr / evalutation_point_count
    return (recall,mrr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='rsc_64', help='dataset name: diginetica/rsc_4/rsc_64/retail')
    parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
    parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--maxLen', type=int, default=40, help='max length of session')
    parser.add_argument('--patience', type=int, default=5, help='patience of bad result')
    parser.add_argument('--dropouts', type=list, default=[0,0.25,0.25], help='patience of bad result')
    parser.add_argument('--windowSize', type=list, default=3, help='size of convolution slip windows')
    parser.add_argument('--penalization', type=bool, default=False, help='if use penalization loss')
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=2,help='the number of steps after which the learning rate decay')


    opt = parser.parse_args()

    PATH=''

    if opt.dataset == 'diginetica':
        path_train = PATH+"narm_diginetica_train.txt"
        path_test = PATH+"narm_diginetica_test.txt"
        num_item = 43098
    elif opt.dataset == 'rsc_64' :
        path_train = PATH+"narm_yo_64_train.txt"
        path_test = PATH+"narm_rsc_64_test.txt"
        num_item = 17695
    elif opt.dataset == 'rsc_4':
        path_train = "narm_rsc_4_train.txt"
        path_test = "narm_rsc_4_test.txt"
        num_item = 30661

    ###load data
    train, valid, test = load_data(path_train,path_test,maxlen=opt.maxLen,sort_by_len=True)
    if valid is not None:
        kf_valid = get_minibatches_idx(len(valid[0]), opt.batchSize)
        print("%d valid examples" % len(valid[0]))
    kf_test = get_minibatches_idx(len(test[0]), opt.batchSize,shuffle=False)
    kf_test = list(kf_test)
    print("%d train examples" % len(train[0]))
    print("%d test examples" % len(test[0]))


    ###model
    model = PACA(batch_size=opt.batchSize,
                 emb_dim=opt.hiddenSize,
                 windowSize=opt.windowSize,
                 num_item=num_item,
                 max_length=opt.maxLen,
                 learning_rate=opt.lr,
                 decay=opt.lr_dc_step * len(train[0]) / opt.batchSize,
                 learningRate_decay=opt.lr_dc,
                 penalization=opt.penalization)
    ###gpu config and session initialize
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    #paras initialize
    history_test = []
    bad_count = 0
    dispFreq = 100

    uidx = 0  # the number of update done
    estop = False  # early stop
    print('Training...')
    for epoch_id in range(opt.epoch):
        n_samples = 0
        epoch_loss = []
        try:
            ###train
            kf = get_minibatches_idx(len(train[0]), opt.batchSize,shuffle=True)
            start_time = time.time()
            for _, train_index in kf:
                uidx += 1

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, lengths, y = prepare_data(x, y) #prepare before epoch
                batch_num = x.shape[1]
                n_samples += batch_num

                dropouts = opt.dropouts
                # print(model.learning_rate)
                outs = sess.run([model.opt_op, model.loss],
                                feed_dict={model.x:x,model.mask:mask,model.labels:y,model.dropouts:dropouts})

                epoch_loss.append(outs[1])

                if np.mod(uidx, dispFreq) == 0:
                    print('Epoch ', epoch_id, 'Update ', uidx, 'Loss ', np.mean(epoch_loss))
                    # print("data_process_time %.2fs , optimal_process_time %.2fs" % (data_process_time,optimal_process_time))
            train_time = time.time()

            ###evaluate
            test_evaluation = forward_evaluate(model, test, kf_test)
            history_test.append(test_evaluation)
            print_metric(test_evaluation)

            if test_evaluation[0][0] > np.array(history_test).max():
                print('Best perfomance updated!')
                bad_count = 0

            if test_evaluation[0][0] < np.array(history_test).max():
                bad_count += 1
                # print('===========================>Bad counter: ' + str(bad_count))
                print('current  recall: ' + str(test_evaluation[0][0]) +
                      '      history max recall:' + str(np.array(history_test).max()))
                if bad_count > opt.patience:
                    estop = True

            end_time = time.time()
            print('Seen %d samples' % n_samples)
            print(('This epoch took %.1fs \ntrain took %.3fs & predict tool %.3fs' % (end_time - start_time,train_time-start_time,end_time-train_time)))
            print('==================================================\n')

            if estop:
                print('Early Stop!')
                break
        except StopIteration:
            pass
    ###output the best result\
    history_test= np.array(history_test)
    ind = np.unravel_index(np.argmax( history_test, axis=None),  history_test.shape)
    best_evaluation = history_test[ind[0]]
    print('=================Best performance=================')
    print_metric(best_evaluation)
    print('==================================================')











