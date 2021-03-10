import argparse
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.semi_supervised import label_propagation
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from keras.models import model_from_yaml
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import  AdaBoostClassifier
from keras import optimizers
import re
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from collections import defaultdict
import pickle
import random
import Extract_fe as data
import matplotlib.pyplot as plt
import pymrmr



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-X', '--Xfile',
        help='File with X data',
        type=str,
        default=None,
        dest='X_filename'
    )

    parser.add_argument(
        '-y', '--yfile',
        help='File with y data',
        type=str,
        default=None,
        dest='y_filename'
    )

    parser.add_argument(
        '-an', '--add_n',
        type=int,
        default=1,
        help='Add n samples from pool per training loop (default: 1)',
        dest='add_n'
    )

    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.7,
        help='Desired threshold for accuracy on test set (default: 0.7)',
        dest='threshold'
    )

    return parser.parse_args()


def train_test_split_rand(X, y, test_size=0.2, seed=None):
    """Split a dataset \\\into training and test at random
    Parameters
    ----------
    X : np.array
        Features of the dataset
    y : np.array
        Labels of the dataset
    test_size : float, optional
        Percentage of data in test set, by default 0.2
    seed : int, optional
        Seed for Random State, by default None
    Returns
    -------
    X_train, X_test, y_train, y_test
        Splitted data
    """

    # X = X.values
    # y = y.values

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # print(X_train.shape)
    return X_train, X_test, y_train, y_test



def calc_performance(y_test, pred_y, test_num,prefix=''):
    """Calculate performance metrics between predicted labels and true labels
    Parameters
    ----------
    y_test : [type]
        [description]
    pred_y : [type]
        [description]
    prefix : str, optional
        [description], by default ''
    Returns
    -------
    [type]
        [description]
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if y_test[index] == 1:
            if y_test[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if y_test[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    # print('tp:', tp, 'fn:', fn, 'tn:', tn, 'fp:', fp)
    acc = (tp + tn) / (tp + tn+fp+fn)
    precision = precision_score(y_true=y_test, y_pred=pred_y, zero_division=0)
    recall=(tp) / (tp + fn)
    f1=2*precision*recall/(precision+recall)
    fpr, tpr, thresholds = roc_curve(y_test, pred_y)  # probas_[:, 1])
    auc = metrics.auc(fpr, tpr)
    # auc=metrics.roc_auc_score(y_test, pred_y)
    # print(auc)
    print(f"{prefix} Accuracy: {acc:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}, F1-score: {f1:.3f}, AUC: {auc:.3f}",'tp:', tp, 'fn:', fn, 'tn:', tn, 'fp:', fp)

    return acc, recall, precision, f1, auc



def rand_sampling(X_pool, y_pool, X_test, y_test, model, model_args, add_n=2, n_init=50, steps=None, verbose=0,
                         threshold=0.7):
    """random selection with pool based sampling by random
    Parameters
    ----------
    X_pool : numpy.array
        Features in pool
    y_pool : numpy.array
        Labels in pool
    X_test : numpy.array
        Features in test
    y_test : numpy.array
        Labels in test
    model : sklearn.model
        Sklearns implementation of a model,
        MLPClassifier,KNeighborsClassifier,AdaBoostClassifier or label_propagation
    model_args : dict
        Argumnts for model function
    add_n : int, optional
        How many samples to add at each iteration, by default 1
    n_init : int, optional
        How many samples from pool are in the training data in the first iteration, by default 10
    steps : int, optional
        Number of iterations to run. If None, then run until all samples from pool have been used or if threshold given, stop when threshold is reached, by default None
    verbose : int, optional
        If 1, print test performance in each iteration, by default 0
    threshold : float, optional
        If given, stop learning system if accuracy score on test set is >= threshold, by default 0.7
    Returns
    -------
    clf : sklearn.model
        Trained classifier
    test_acc, test_recall, test_precision, test_f1: lists
        Performance measures on test set at each iteration
    """

    # mix up order of pool indexes
    order = np.random.permutation(range(len(X_pool)))

    # initialize poolidxs
    poolidxs = np.arange(len(X_pool))

    # take n_init samples from pool as training set
    # print(order)
    trainset = order[:n_init]
    # print(trainset)
    X_train = X_pool[trainset]
    y_train = y_pool[trainset]

    print(model)
    # remove the first n_init idxs from poolidxs
    poolidxs = np.setdiff1d(poolidxs, trainset)

    # initialize model
    clf = model(**model_args)

    if steps is None:
        steps = len(poolidxs) // 1



    # training loop
    test_acc, test_recall, test_precision, test_f1, AUC = [], [], [], [],[]
    for i in range(steps):
        count = 0
        # fit model
        clf.fit(X_train, y_train.ravel())#.ravel()

        # calculate performance on test set
        y_pred = clf.predict(X_test)


        test_num=len(X_test)
        acc, recall, precision, f1, auc = calc_performance(y_test=y_test, pred_y=y_pred,test_num=test_num)
        test_acc.append((len(X_train), acc))
        # test_recall.append((len(X_train), recall))
        test_precision.append((len(X_train), precision))
        # test_f1.append((len(X_train), f1))

        # calculate label probabilities for samples remaining in pool
        y_prob = clf.predict_proba(X_pool[poolidxs])

        new_order = np.random.permutation(range(len(y_prob)))
        new_idx = new_order[:add_n]

        X_add = X_pool[new_idx]
        y_add = y_pool[new_idx]

        X_train = np.concatenate((
            X_train,
            X_add
        ))
        y_train = np.concatenate((
            y_train,
            y_add
        ))


        # remove from pool
        poolidxs = np.setdiff1d(poolidxs, new_idx)

        if verbose == 1:
            print(f"Step {i + 1}/{steps}: Test accuracy: {acc:.3f}", end='\r')

        if threshold is not None and acc >= threshold:
            print("Desired accuracy reached. Stopping training.")
            break
    # if steps % 3 == 0:

        # np.savetxt("result.txt", X_train)
        count+=1
        # show(X_train, y_train, X_test, y_pred,count)

    return clf, test_acc, test_recall, test_precision, test_f1


def train(X, y, split_func, sampling_func, add_n, steps=30, model=label_propagation.LabelSpreading, model_args={}, split_args={}):


    X_train, X_test, y_train, y_test = split_func(X, y, **split_args)

    _, test_acc, test_recall, test_precision, test_f1 = rand_sampling(
        X_pool=X_train,
        X_test=X_test,
        y_pool=y_train,
        y_test=y_test,
        model=model,
        model_args=model_args,
        verbose=1,
        add_n=add_n,
        n_init=add_n,
        threshold=None,
        steps=30
    )

    return test_acc, test_recall, test_precision, test_f1


def dd():
    return defaultdict(dict)


def show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels,count):
    plt.style.use('ggplot')
    for i in range(Mat_Label.shape[0]):

        if int(labels[i]) == 0:
            line1=plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Dm',markersize=4,)

        elif int(labels[i]) == 1:
            line2=plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Db',markersize=4)
            # plt.legend()
        # else:
        #     plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Dy')

    for i in range(Mat_Unlabel.shape[0]):
        if int(unlabel_data_labels[i]) == 0:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'om',markersize=4,)
            # plt.legend()
        elif int(unlabel_data_labels[i]) == 1:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'ob',markersize=4)
            # plt.legend()
        # else:
        #     plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'oy',markersize=4)


    plt.xlabel('X1')
    plt.ylabel('X2')
    # plt.legend(handles=[line1,line2,line3,line4],labels=['label(0)', 'label(1)','ulabel(0)','ulabel(1)'])

    plt.xlim(-0.05, 0.25)
    plt.ylim(-0.07, 0.05)
    plt.rcParams['savefig.dpi'] = 300
    # plt.savefig("C:\\Users\\86151\\Desktop\\2021\\picture\\san\\temp{}.jpg".format(count))
    # plt.savefig("D:/figures/temp{}.png".format(i))

    plt.show()

if __name__ == "__main__":
    args = parse_args()

    # load files

    label, pto = data.read_sequence()

    aac = data.fe()

    dpc = data.DPC()
    # pca = PCA(n_components=100)
    # pca = pca.fit(dpc)
    # dpc= pca.transform(dpc)
    # dpc= np.array(dpc)

    # gaac=data.gaac()

    pcp = data.PC_PseAAC()

    bpf = data.get_bpf()



    X = np.concatenate((bpf,pcp,aac,dpc), axis=1)
    # X=pcp
    # print(X.shape)
    # pca = PCA(n_components=300)
    # pca = pca.fit(X)
    # X = pca.transform(X)
    # X = np.array(X)

    y = label

    X=np.array(X,dtype=np.float32)
    y=np.array(y,dtype=np.float32)


    # parameter settings for classifiers
    models = {

        'Neural Network': {
            'model': MLPClassifier,
            'model_args': {
                'max_iter': 1000,
                 'random_state' :0
            }
        },
        'AdaBoostClassifier':{
            'model':AdaBoostClassifier,
            'model_args': {
                'n_estimators': 200,
                'learning_rate': 0.01,
                'random_state': 100

            }
        },
        'label_propagation':{
            'model':label_propagation.LabelSpreading,
            'model_args': {
                'gamma':0.25,
                'max_iter':15
                }
        },

        'KNeighborsClassifier':{
            'model':KNeighborsClassifier,
            'model_args':{
                'n_neighbors' : 8

            }
        }

    }

    # functions and settings for data splitters
    split_funcs = [train_test_split_rand]
    split_args = {
        'Random': [{}],
    }
    split_keys = ['Random']

    # dict to store results in
    res = defaultdict(dd)

    # test each combination
    for model_name, model_settings in models.items():
        for split_func, split_key in zip(split_funcs, split_keys):
            for split_arg in split_args[split_key]:

                split_label = split_key
                if 'Random positive sampling' == split_key:
                    split_label = split_key.replace('Random', f"{split_arg['pos_frac'] * 100}%")

                for i in range(15):  # repeat 15 times
                    print(f"{model_name} {split_label} - iteration: {i + 1}/5")

                    # set random seed
                    random.seed()

                    test_acc, test_recall, test_precision, test_f1 = train(
                        X=X,
                        y=y,
                        split_func=split_func,
                        sampling_func=rand_sampling,
                        add_n=args.add_n,
                        model=model_settings['model'],
                        model_args=model_settings['model_args'],
                        split_args=split_arg
                    )

                    res[model_name][split_label][i] = {
                        # 'test_acc': test_acc,
                        'test_recall': test_recall,
                        'test_precision': test_precision,
                        'test_f1': test_f1
                    }

    # dump result dict in a pickled file
    with open('results.pkl', 'wb') as dest:
        pickle.dump(res, dest)