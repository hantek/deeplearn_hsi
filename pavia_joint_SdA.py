__author__ = "Zhouhan LIN"
__date__ = "June 2013"
__version__ = "1.0"

import os
import time
import pdb
import scipy.io as sio
import numpy
import scipy
import theano
import theano.tensor as T
from scipy.stats import t
from sklearn import svm
from sklearn.metrics import confusion_matrix
from theano.tensor.shared_randomstreams import RandomStreams
import sys

from SdA import SdA
from hsi_utils import *

cmap = numpy.asarray( [[0, 0, 0],               
                       [192, 192, 192],         
                       [0, 255, 0],
                       [0, 255, 255],
                       [0, 128, 0],
                       [255, 0, 255],
                       [165, 82, 41],
                       [128, 0, 128],
                       [255, 0, 0],
                       [255, 255, 0]], dtype='int32')

def run_sda(datasets=None, batch_size=100,
            window_size=7, n_principle=3,
            pretraining_epochs=2000, pretrain_lr=0.02,
            training_epochs=10000,  finetune_lr=0.008, 
            hidden_layers_sizes=[310, 100], corruption_levels = [0., 0.]):
    """
    This function maps spatial PCs to a deep representation.
    
    Parameters:
    datasets:           A list containing 3 tuples. Each tuple have 2 entries, 
                        which are theano.shared variables. They stands for train,
                        valid, test data.
    batch_size:         Batch size.
    pretraining_epochs: Pretraining epoches.
    pretrain_lr:        Pretraining learning rate.
    training_epochs:    Fine-tuning epoches.
    finetune_lr:        Fine-tuning learning rate.
    hidden_layers_sizes:A list containing integers. Each intger specifies a size
                        of a hidden layer.
    corruption_levels:  A list containing floats in the inteval [0, 1]. Each 
                        number specifies the corruption level of its corresponding
                        hidden layer.

    Return:
    spatial_rep:        2-D numpy.array. Deep representation for each spatial sample.
    test_score:         Accuracy this representations yield on the trained SdA.
    """
    
    print 'finetuning learning rate=', finetune_lr
    print 'pretraining learning rate=', pretrain_lr
    print 'pretraining epoches=', pretraining_epochs
    print 'fine tuning epoches=', training_epochs
    print 'batch size=', batch_size
    print 'hidden layers sizes=', hidden_layers_sizes
    print 'corruption levels=', corruption_levels

    # compute number of minibatches for training, validation and testing
    n_train_batches = datasets[0][0].get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = SdA(numpy_rng=numpy_rng, n_ins=datasets[0][0].get_value(borrow=True).shape[1],
              hidden_layers_sizes=hidden_layers_sizes,
              n_outs=gnd_img.max())

    ################################################################################
                               # PRETRAINING THE MODEL #
                               #########################

    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=datasets[0][0],
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            
            if epoch % 100 == 0:
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c)

    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ################################################################################
                                # FINETUNING THE MODEL #
                                ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets, batch_size=batch_size,
        learning_rate=finetune_lr)

    print '... finetunning the model'
    # early-stopping parameters
    patience = 100 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(10 * n_train_batches, patience / 2)
                            # go through this many
                            # minibatche before checking the network
                            # on the validation set; in this case we
                            # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                (best_validation_loss * 100., test_score * 100.))
    print >> sys.stdout, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    # keep the following line consistent with line 227, function "prepare_data"
    filename = 'pavia_l1sda_pt%d_ft%d_lrp%.4f_f%.4f_bs%d_pca%d_ws%d' % \
                (pretraining_epochs, training_epochs, pretrain_lr, finetune_lr, 
                 batch_size, n_principle, window_size) 

    print '... saving parameters'
    sda.save_params(filename + '_params.pkl')

    print '... classifying test set with learnt model:'
    pred_func = theano.function(inputs=[sda.x], outputs=sda.logLayer.y_pred)
    pred_test = pred_func(datasets[2][0].get_value(borrow=True))
    true_test = datasets[2][1].get_value(borrow=True)
    true_valid = datasets[1][1].get_value(borrow=True)
    true_train = datasets[0][1].get_value(borrow=True)
    result_analysis(pred_test, true_train, true_valid, true_test)

    print '... classifying the whole image with learnt model:'
    print '...... extracting data'
    data_spectral, data_spatial, _, _ = \
        T_pca_constructor(hsi_img=img, gnd_img=gnd_img, n_principle=n_principle, 
                          window_size=window_size, flag='unsupervised', 
                          merge=True)
    
    start_time = time.clock()
    print '...... begin '
    y = pred_func(data_spectral) + 1
    print '...... done '
    end_time = time.clock()
    print 'finished, running time:%fs' % (end_time - start_time)

    y_rgb = cmap[y, :]
    margin = (window_size / 2) * 2  # floor it to a multiple of 2
    y_image = y_rgb.reshape(width - margin, height - margin, 3)
    scipy.misc.imsave(filename + 'wholeimg.png' , y_image)
    print 'Saving classification results'
    sio.savemat(filename + 'wholeimg.mat', 
                {'y': y.reshape(width - margin, height - margin)})
    
    ############################################################################
    print '... performing Student\'s t-test'
    best_c = 10000.
    best_g = 10.
    svm_classifier = svm.SVC(C=best_c, gamma=best_g, kernel='rbf')
    svm_classifier.fit(datasets[0][0].get_value(), datasets[0][1].get_value())

    data = [numpy.vstack((datasets[1][0].get_value(),
                          datasets[2][0].get_value())),
            numpy.hstack((datasets[1][1].get_value(),
                          datasets[2][1].get_value()))]
    numpy_rng = numpy.random.RandomState(89677)
    num_test = 100
    print 'Total number of tests: %d' % num_test
    k_sae = []
    k_svm = []
    for i in xrange(num_test):
        [_, _], [_, _], [test_x, test_y], _ = \
        train_valid_test(data, ratio=[0, 1, 1], batch_size=1, 
                         random_state=numpy_rng.random_integers(1e10))
        test_y = test_y + 1 # fix the label scale problem
        pred_y = pred_func(test_x)
        cm = confusion_matrix(test_y, pred_y)
        pr_a = cm.trace()*1.0 / test_y.size
        pr_e = ((cm.sum(axis=0)*1.0/test_y.size) * \
                (cm.sum(axis=1)*1.0/test_y.size)).sum()
        k_sae.append( (pr_a - pr_e) / (1 - pr_e) )

        pred_y = svm_classifier.predict(test_x)
        cm = confusion_matrix(test_y, pred_y)
        pr_a = cm.trace()*1.0 / test_y.size
        pr_e = ((cm.sum(axis=0)*1.0/test_y.size) * \
                (cm.sum(axis=1)*1.0/test_y.size)).sum()
        k_svm.append( (pr_a - pr_e) / (1 - pr_e) )

    std_k_sae = numpy.std(k_sae)
    std_k_svm = numpy.std(k_svm)
    mean_k_sae = numpy.mean(k_sae)
    mean_k_svm = numpy.mean(k_svm)
    left =    ( (mean_k_sae - mean_k_svm) * numpy.sqrt(num_test*2-2)) \
            / ( numpy.sqrt(2./num_test) * num_test * (std_k_sae**2 + std_k_svm**2) )

    rv = t(num_test*2.0 - 2)
    right = rv.ppf(0.95)

    print '\tstd\t\tmean'
    print 'k_sae\t%f\t%f' % (std_k_sae, mean_k_sae)
    print 'k_svm\t%f\t%f' % (std_k_svm, mean_k_svm)
    if left > right:
        print 'left = %f, right = %f, test PASSED.' % (left, right)
    else:
        print 'left = %f, right = %f, test FAILED.' % (left, right)
    
    
    return test_score
    
if __name__ == '__main__':
    print '... loanding data'
    hsi_file = u'/home/hantek/data/hsi_data/pavia/PaviaU.mat'
    gnd_file = u'/home/hantek/data/hsi_data/pavia/PaviaU_gt.mat'
    data = sio.loadmat(hsi_file)
    img = scale_to_unit_interval(data['paviaU'].astype(theano.config.floatX))
    width = img.shape[0]
    height = img.shape[1]
    bands = img.shape[2]
    data = sio.loadmat(gnd_file)
    gnd_img = data['paviaU_gt'].astype(numpy.int32)

    print '... extracting train-valid-test sets'
    datasets, _, _, _ = \
        prepare_data(hsi_img=img, gnd_img=gnd_img, merge=True, 
                     window_size=7, n_principle=4, batch_size=50)

    print '... Running hybrid feature extraction on SdA'
    spatial_accuracy = run_sda(datasets=datasets, batch_size=50, 
                               window_size=7, n_principle=4,
                               pretraining_epochs=500, pretrain_lr=0.5,
                               training_epochs=80000,  finetune_lr=0.03,
                               hidden_layers_sizes=[280, 100, 100, 100], 
                               corruption_levels = [0., 0., 0., 0.])

