#!/usr/bin/python
#

import pdb
import time
import sys
import scipy.io as sio
import numpy
import theano
import pylab as pl
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

#-------------------------------------------------------------------------------
"""
The scale_to_unit_interval() and tile_raster_images() functions are from the 
Deep Learning Tutorial repo:
    
    https://github.com/lisa-lab/DeepLearningTutorials

Below are the corresponding licence.

LICENSE
=======

Copyright (c) 2010--2015, Deep Learning Tutorials Development Team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Theano nor the names of its contributors may be
      used to endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


#-------------------------------------------------------------------------------
def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


#-------------------------------------------------------------------------------
# the following color map is for generating wholeimage classification figures.
cmap = numpy.asarray( [[0, 0, 0],               
                       [0, 205, 0],         
                       [127, 255, 0],       
                       [46, 139, 87],
                       [0, 139, 0],
                       [160, 82, 45],
                       [0, 255, 255],
                       [255, 255, 255],
                       [216, 191, 216],
                       [255, 0, 0],
                       [139, 0, 0],
                       [0, 0, 255],
                       [255, 255, 0],
                       [238, 154, 0],
                       [85, 26, 139],
                       [255, 127, 80]], dtype='int32')

#-------------------------------------------------------------------------------
def result_analysis(prediction, train_truth, valid_truth, test_truth, 
                    verbose=False):
    assert prediction.shape == test_truth.shape
    print "Detailed information in each category:"
    print "                          Number of Samples"
    print "Class No.       TRAIN   VALID   TEST    RightCount   RightRate"
    for i in xrange(test_truth.min(), test_truth.max()+1):
        right_prediction = ( (test_truth-prediction) == 0 )
        right_count = numpy.sum(((test_truth==i) * right_prediction)*1)
        print "%d\t\t%d\t%d\t%d\t%d\t%f" % \
            (i, 
             numpy.sum((train_truth==i)*1), 
             numpy.sum((valid_truth==i)*1), 
             numpy.sum((test_truth==i)*1),
             right_count,
             right_count * 1.0 / numpy.sum((test_truth==i)*1)
            )
    
    total_right_count = numpy.sum(right_prediction*1)
    print "Overall\t\t%d\t%d\t%d\t%d\t%f" % \
            (train_truth.size, 
             valid_truth.size, 
             test_truth.size, 
             total_right_count,
             total_right_count * 1.0 / test_truth.size
            )
    
    cm = confusion_matrix(test_truth, prediction)
    pr_a = cm.trace()*1.0 / test_truth.size
    pr_e = ((cm.sum(axis=0)*1.0/test_truth.size) * \
            (cm.sum(axis=1)*1.0/test_truth.size)).sum()
    k = (pr_a - pr_e) / (1 - pr_e)
    print "kappa index of agreement: %f" % k
    print "confusion matrix:"
    print cm
    # Show confusion matrix
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    if verbose:
        pl.show()
    else:
        filename = 'conf_mtx_' + str(time.time()) + '.png'
        pl.savefig(filename)

#-------------------------------------------------------------------------------
def PCA_tramsform_img(img=None, n_principle=3):
    """
    This function trainsforms an HSI by 1-D PCA. PCA is fitted on the whole data
    and is conducted on the spectral dimension, rendering the image from size 
    length * width * dim to length * width * n_principle. 
    
    Parameters:
    img:                initial unregularizaed HSI.
    n_principle:        Target number of principles we want.
    
    Return:
    reg_img:            Regularized, transformed image.
    
    WARNNING: RELATIVE ENERGY BETWEEN PRINCIPLE COMPONENTS CHANGED IN THIS 
    IMPLEMENTATION. YOU MAY NEED TO ADD PENALTY MULTIPLIERS IN THE HIGHER NETWORKS
    TO REIMBURSE IT.
    """
    length = img.shape[0]
    width = img.shape[1]
    dim = img.shape[2]
    # reshape img, HORIZONTALLY strench the img, without changing the spectral dim.
    reshaped_img = numpy.asarray(img.reshape(length*width, dim), 
                                 dtype=theano.config.floatX)
    pca = PCA(n_components=n_principle)
    pca_img = pca.fit_transform(reshaped_img)
    
    # Regularization: Think about energy of each principles here.
    reg_img = scale_to_unit_interval(ndar=pca_img, eps=1e-8)
    reg_img = numpy.asarray(reg_img.reshape(length, width, n_principle), 
                            dtype=theano.config.floatX)
    energy_dist = pca.explained_variance_ratio_
    residual = 1 - numpy.sum(energy_dist[0: n_principle])
    return reg_img, energy_dist, residual

#-------------------------------------------------------------------------------
def T_pca_constructor(hsi_img=None, gnd_img=None, n_principle=3, window_size=1, 
                      flag='supervised', merge=False):
    """
    This function constructs the spectral and spatial facade for each training 
    pixel. 
    
    Spectral data returned are simply spectra.
    
    spatial data returned are the former n_principle PCs of a neibor region
    around each extracted pixel. Size of the neibor region is determined by 
    window_size. And all the values for a pixel are flattened to 1-D size. So 
    data_spatial is finally a 2-D numpy.array.
    
    All the returned data are regularized to [0, 1]. Set window_size=1 to
    get pure spectral returnings.

    Parameters:
    hsi_img=None:       3-D numpy.ndarray, dtype=float, storing initial 
                        hyperspectral image data.
    gnd_img=None:       2-D numpy.ndarray, dtype=int, containing tags for pixeles.
                        The size is the same to the hsi_img size, but with only 
                        1 band.
    n_principle:        Target number of principles we want to consider in the 
                        spatial infomation.
    window_size:        Determins the scale of spatial information incorporated. 
                        Must be odd.
    flag:               For 'unsupervised', all possible pixels except marginals
                        are processed. For 'supervised', only pixels with 
                        non-zero tags are processed.
    
    Return:
    data_spectral:      2-D numpy.array, sized (sample number) * (band number). 
                        Consists of regularized spectra for all extracted pixels.
    data_spatial:       2-D numpy.array, sized (sample number) * (window_size^2).
                        Consists the former n_principle PCs of a neibor region
                        around each extracted pixel. Size of the neibor region
                        is determined by window_size.
    gndtruth:           1-D numpy.array, sized (sample number) * 1. Truth value 
                        for each extracted pixel. 
    extracted_pixel_ind:2-D numpy.array, sized (length) * (width). Indicating 
                        which pixels are selected.
    """
    # PCA transformation
    pca_img, _, _ = PCA_tramsform_img(img=hsi_img, n_principle=n_principle)

    # Regularization
    hsi_img = scale_to_unit_interval(ndar=hsi_img, eps=1e-8)
    length = hsi_img.shape[0]
    width = hsi_img.shape[1]
    dim = hsi_img.shape[2]

    # reshape img, HORIZONTALLY strench the img, without changing the spectral dim.
    reshaped_img = numpy.asarray(hsi_img.reshape(length*width, dim), 
                                 dtype=theano.config.floatX)
    reshaped_gnd = gnd_img.reshape(gnd_img.size) 
    
    # mask ensures marginal pixels eliminated, according to window_size
    threshold = (window_size-1) / 2
    if window_size >= 1 and window_size < width-1 and window_size < length-1:
        mask_false = numpy.array([False, ] * width)
        mask_true = numpy.hstack((numpy.array([False, ] * threshold, dtype='bool'),
                                  numpy.array([True, ] * (width-2*threshold)), 
                                  numpy.array([False, ] * threshold, dtype='bool')))
        mask = numpy.vstack((numpy.tile(mask_false, [threshold, 1]), 
                             numpy.tile(mask_true, [length-2*threshold, 1]), 
                             numpy.tile(mask_false, [threshold, 1])))
        reshaped_mask = mask.reshape(mask.size)
    else:
        print >> sys.stderr, ('window_size error. choose 0 < window_size < width-1')
        
    # construct groundtruth, and determine which pixel to process
    if flag == 'supervised':
        extracted_pixel_ind = (reshaped_gnd > 0) * reshaped_mask
        gndtruth = reshaped_gnd[extracted_pixel_ind]
        extracted_pixel_ind = numpy.arange(reshaped_gnd.size)[extracted_pixel_ind]
    elif flag == 'unsupervised':
        extracted_pixel_ind = numpy.arange(reshaped_gnd.size)[reshaped_mask]
        gndtruth = numpy.array([], dtype='int')
    else:
        print >> sys.stderr, ('\"flag\" parameter error. ' +
                              'What type of learning you are doing?')
        return
    
    # construct data_spectral
    data_spectral = reshaped_img[extracted_pixel_ind, :]

    # construct data_spatial
    if window_size == 1:
        data_spatial = numpy.array([])
    else:
        data_spatial = numpy.zeros([extracted_pixel_ind.size, 
                                    window_size * window_size * n_principle], 
                                   dtype=theano.config.floatX)
        i = 0
        for ipixel in extracted_pixel_ind:
            ipixel_h = ipixel % width
            ipixel_v = ipixel / width
            data_spatial[i, :] = \
            pca_img[ipixel_v-threshold : ipixel_v+threshold+1, 
                    ipixel_h-threshold : ipixel_h+threshold+1, :].reshape(
                   window_size*window_size*n_principle)
            i += 1
    
    # if we want to merge data, merge it
    if merge:
        data_spectral = numpy.hstack((data_spectral, data_spatial))
    
    return data_spectral, data_spatial, gndtruth, extracted_pixel_ind

#-------------------------------------------------------------------------------
def train_valid_test(data, ratio=[6, 2, 2], batch_size=50, random_state=None):
    """
    This function splits data into three parts, according to the "ratio" parameter
    given in the lists indicating training, validating, testing data ratios.
    data:             a list containing:
                      1. A 2-D numpy.array object, with each patterns listed in 
                         ROWs. Input data dimension MUST be larger than 1.
                      2. A 1-D numpy.array object, tags for each pattern. 
                         '0' indicates that the tag for the corrresponding 
                         pattern is unknown.
    ratio:            A list having 3 elements, indicating ratio of training, 
                      validating and testing data ratios respectively.
    batch_size:       bathc_size helps to return an appropriate size of training
                      samples, which has divisibility over batch_size. 
                      NOTE: batch_size cannot be larger than the minimal size of
                      all the trainin, validate and test dataset!
    random_state:     If we give the same random state and the same ratio on the
                      same data, the function will yield a same split for each
                      function call.
    return:
    [train_data_x, train_data_y]: 
    [valid_data_x, valid_data_y]:
    [test_data_x , test_data_y ]:
                      Lists containing 2 numpy.array object, first for data and 
                      second for truth. They are for training, validate and test
                      respectively. All the tags are integers in the range 
                      [0, data[1].max()-1]. 
    split_mask
    """

    rand_num_generator = numpy.random.RandomState(random_state)

    #---------------------------split dataset-----------------------------------
    random_mask = rand_num_generator.random_integers(1, sum(ratio), data[0].shape[0])
    split_mask = numpy.array(['tests', ] * data[0].shape[0])
    split_mask[random_mask <= ratio[0]] = 'train'
    split_mask[(random_mask <= ratio[1]+ratio[0]) * (random_mask > ratio[0])] = 'valid'
    
    train_data_x = data[0][split_mask == 'train', :]
    train_data_y = data[1][split_mask == 'train']-1
    valid_data_x = data[0][split_mask == 'valid', :]
    valid_data_y = data[1][split_mask == 'valid']-1
    test_data_x  = data[0][split_mask == 'tests', :]
    test_data_y  = data[1][split_mask == 'tests']-1
    '''
    #---------------------------split dataset-----------------------------------
    features, tags = data
    n_classes = tags.max()
    index = numpy.arange(tags.size)
    ratio_sum = sum(ratio)
    
    split_mask = numpy.array(['tests', ] * data[0].shape[0])

    test_features  = []
    valid_features = []
    train_features = []
    test_tags  = []
    valid_tags = []
    train_tags = []
    
    for iclass in xrange(1, n_classes+1):
        # pdb.set_trace()
        itag_ind = index[tags == iclass]
        itag_ind = rand_num_generator.permutation(itag_ind)
        
        itest_count  = \
        numpy.round(1.0*itag_ind.size * ratio[2] / ratio_sum).astype(tags.dtype)
        ivalid_count = \
        numpy.round(1.0*itag_ind.size * ratio[1] / ratio_sum).astype(tags.dtype)
        
        itest_ind  = itag_ind[:itest_count]
        ivalid_ind = itag_ind[itest_count:(itest_count+ivalid_count)]
        itrain_ind = itag_ind[(itest_count+ivalid_count):]
        
        itest_features  = features[itest_ind,:]
        ivalid_features = features[ivalid_ind,:]
        itrain_features = features[itrain_ind,:]
        test_features.append(itest_features)
        valid_features.append(ivalid_features)
        train_features.append(itrain_features)
        
        itest_tags  = tags[itest_ind,:]
        ivalid_tags = tags[ivalid_ind,:]
        itrain_tags = tags[itrain_ind,:]
        test_tags.append(itest_tags)
        valid_tags.append(ivalid_tags)
        train_tags.append(itrain_tags)
        
        split_mask[ivalid_ind] = 'valid'
        split_mask[itrain_ind] = 'train'
        
    test_features  = numpy.vstack(test_features)
    valid_features = numpy.vstack(valid_features)
    train_features = numpy.vstack(train_features)
    test_tags  = numpy.hstack(test_tags)
    valid_tags = numpy.hstack(valid_tags)
    train_tags = numpy.hstack(train_tags)
    
    # random permute the tags
    index = rand_num_generator.permutation(numpy.arange(test_tags.size))
    test_data_y = test_tags[index]
    test_data_x = test_features[index,:]
    index = rand_num_generator.permutation(numpy.arange(valid_tags.size))
    valid_data_y = valid_tags[index]
    valid_data_x = valid_features[index,:]
    index = rand_num_generator.permutation(numpy.arange(train_tags.size))
    train_data_y = train_tags[index]
    train_data_x = train_features[index,:]
    #---------------------------------------------------------------------------
    '''
    # tackle the batch size mismatch problem
    mis_match = train_data_x.shape[0] % batch_size
    if mis_match != 0:
        mis_match = batch_size - mis_match
        train_data_x = numpy.vstack((train_data_x, train_data_x[0:mis_match, :]))
        train_data_y = numpy.hstack((train_data_y, train_data_y[0:mis_match]))
    
    mis_match = valid_data_x.shape[0] % batch_size
    if mis_match != 0:
        mis_match = batch_size - mis_match
        valid_data_x = numpy.vstack((valid_data_x, valid_data_x[0:mis_match, :]))
        valid_data_y = numpy.hstack((valid_data_y, valid_data_y[0:mis_match]))
    
    mis_match = test_data_x.shape[0] % batch_size
    if mis_match != 0:
        mis_match = batch_size - mis_match
        test_data_x = numpy.vstack((test_data_x, test_data_x[0:mis_match, :]))
        test_data_y = numpy.hstack((test_data_y, test_data_y[0:mis_match]))
    
    return [train_data_x, train_data_y], \
           [valid_data_x, valid_data_y], \
           [test_data_x , test_data_y], split_mask

#-------------------------------------------------------------------------------
def prepare_data(hsi_img=None, gnd_img=None, window_size=7, n_principle=3, 
                 batch_size=50, merge=False, ratio=[6, 2, 2]):
    """
    Process the data from file path to splited train-valid-test sets; Binded in 
    dataset_spectral and dataset_spatial respectively.
    
    Parameters:
    
    hsi_img=None:       3-D numpy.ndarray, dtype=float, storing initial 
                        hyperspectral image data.
    gnd_img=None:       2-D numpy.ndarray, dtype=int, containing tags for pixeles.
                        The size is the same to the hsi_img size, but with only 
                        1 band.
    window_size:        Size of spatial window. Pass an integer 1 if no spatial 
                        infomation needed.
    n_principle:        This many principles you want to incorporate while 
                        extracting spatial info.
    merge:              If merge==True, the returned dataset_spectral has 
                        dataset_spatial stacked in the tail of it; else if 
                        merge==False, the returned dataset_spectral and 
                        dataset_spatial will have spectral and spatial information
                        only, respectively.
    Return:
    
    dataset_spectral:   
    dataset_spatial:    
    extracted_pixel_ind:
    split_mask:
    """
    # gnd_mask:           A matrix the same size to ground matrix, containing flags
    #                     about pixel usage:
    #                         0 for NOT chozen;
    #                         1 for chozen as training;
    #                         2 for chozen as validation;
    #                         3 for chozen as testing;

    data_spectral, data_spatial, gndtruth, extracted_pixel_ind = \
        T_pca_constructor(hsi_img=hsi_img, gnd_img=gnd_img, n_principle=n_principle,
                          window_size=window_size, flag='supervised')
    
    ################ separate train, valid and test spatial data ###############
    [train_spatial_x, train_y], [valid_spatial_x, valid_y], [test_spatial_x, test_y], split_mask = \
        train_valid_test(data=[data_spatial, gndtruth], ratio=ratio,
                         batch_size=batch_size, random_state=123)

    # convert them to theano.shared values
    train_set_x = theano.shared(value=train_spatial_x, name='train_set_x', borrow=True)
    valid_set_x = theano.shared(value=valid_spatial_x, name='valid_set_x', borrow=True)
    test_set_x  = theano.shared(value=test_spatial_x,  name='test_set_x',  borrow=True)
    train_set_y = theano.shared(value=train_y, name='train_set_y', borrow=True)
    valid_set_y = theano.shared(value=valid_y, name='valid_set_y', borrow=True)
    test_set_y  = theano.shared(value=test_y,  name='test_set_y',  borrow=True)
    dataset_spatial = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), 
                       (test_set_x, test_set_y)]

    ############### separate train, valid and test spectral data ###############
    [train_spectral_x, train_y], [valid_spectral_x, valid_y], [test_spectral_x, test_y], split_mask = \
        train_valid_test(data=[data_spectral, gndtruth], ratio=ratio,
                         batch_size=batch_size, random_state=123)
    
    # if we want to merge data, merge it
    if merge:
        train_spectral_x = numpy.hstack((train_spectral_x, train_spatial_x))
        valid_spectral_x = numpy.hstack((valid_spectral_x, valid_spatial_x))
        test_spectral_x  = numpy.hstack((test_spectral_x,  test_spatial_x))
    
    # convert them to theano.shared values
    train_set_x = theano.shared(value=train_spectral_x, name='train_set_x', borrow=True)
    valid_set_x = theano.shared(value=valid_spectral_x, name='valid_set_x', borrow=True)
    test_set_x  = theano.shared(value=test_spectral_x,  name='test_set_x',  borrow=True)
    train_set_y = theano.shared(value=train_y, name='train_set_y', borrow=True)
    valid_set_y = theano.shared(value=valid_y, name='valid_set_y', borrow=True)
    test_set_y  = theano.shared(value=test_y,  name='test_set_y',  borrow=True)
    dataset_spectral = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), 
                        (test_set_x, test_set_y)]
    
    return dataset_spectral, dataset_spatial, extracted_pixel_ind, split_mask

if __name__ == '__main__':
    """ Sample usage. """
    print '... Testing function result_analysis'
    import random
    from sklearn import svm, datasets
    
    # import some data to play with
    print '... loading Iris data'
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    n_samples, n_features = X.shape
    p = range(n_samples)
    random.seed(0)
    random.shuffle(p)
    X, y = X[p], y[p]
    half = int(n_samples / 2)

    # Run classifier
    print '... classifying'
    classifier = svm.SVC(kernel='linear')
    y_ = classifier.fit(X[:half], y[:half]).predict(X[half:])

    result_analysis(y_, y[:half], numpy.asarray([]), y[half:])
    
    # load .mat files
    print '... loading KSC data'
    hsi_file = u'/home/hantek/data/hsi_data/kennedy/Kennedy_denoise.mat'
    gnd_file = u'/home/hantek/data/hsi_data/kennedy/Kennedy_groundtruth.mat'
    
    data = sio.loadmat(hsi_file)
    img = numpy.float_(data['Kennedy176'])

    data = sio.loadmat(gnd_file)
    gnd_img = data['Kennedy_groundtruth']
    gnd_img = gnd_img.astype(numpy.int32)

    #-for debug use-------------------------------------------------------------
    # pca_img, _, residual = PCA_tramsform_img(img=img, n_principle=3)
    # print residual
    # sio.savemat('pca_img.mat', {'pca_ksc_img': pca_img})
    # img = numpy.arange(300, dtype=theano.config.floatX).reshape(10, 10, 3)
    # gnd_img = numpy.zeros((10, 10), dtype='int32')
    # gnd_img[[1, 2, 3, 7, 5], [1, 3, 5, 7, 7]] = 2
    #---------------------------------------------------------------------------

    print '... spliting train-valid-test sets'
    dataset_spectral, dataset_spatial, extracted_pixel_ind, split_mask = \
        prepare_data(hsi_img=img, gnd_img=gnd_img, window_size=7, n_principle=3, batch_size=50, merge=True)

    if raw_input('Spliting finished. Do you want to check the data (Y/n)? ') == 'Y':
        spectral_train_x = dataset_spectral[0][0].get_value()
        spectral_train_y = dataset_spectral[0][1].get_value()
        spectral_valid_x = dataset_spectral[1][0].get_value()
        spectral_valid_y = dataset_spectral[1][1].get_value()
        spectral_test_x  = dataset_spectral[2][0].get_value()
        spectral_test_y  = dataset_spectral[2][1].get_value()
                                 
        spatial_train_x  = dataset_spatial[0][0].get_value()
        spatial_train_y  = dataset_spatial[0][1].get_value()
        spatial_valid_x  = dataset_spatial[1][0].get_value()
        spatial_valid_y  = dataset_spatial[1][1].get_value()
        spatial_test_x   = dataset_spatial[2][0].get_value()
        spatial_test_y   = dataset_spatial[2][1].get_value()

        print 'shape of:' 
        print 'spectral_train_x: \t', 
        print spectral_train_x.shape, 
        print 'spectral_train_y: \t', 
        print spectral_train_y.shape
        print 'spectral_valid_x: \t', 
        print spectral_valid_x.shape,
        print 'spectral_valid_y: \t', 
        print spectral_valid_y.shape
        print 'spectral_test_x: \t', 
        print spectral_test_x.shape,
        print 'spectral_test_y: \t', 
        print spectral_test_y.shape
        
        print 'spatial_train_x: \t', 
        print spatial_train_x.shape,
        print 'spatial_train_y: \t', 
        print spatial_train_y.shape
        print 'spatial_valid_x: \t', 
        print spatial_valid_x.shape, 
        print 'spatial_valid_y: \t', 
        print spatial_valid_y.shape
        print 'spatial_test_x: \t', 
        print spatial_test_x.shape, 
        print 'spatial_test_y: \t', 
        print spatial_test_y.shape
        
        print 'total tagged pixel number: %d' % extracted_pixel_ind.shape[0]
        print 'split_mask shape: %d' % split_mask.shape
        
        print '... checking tags in spatial and spectral data'
        trainset_err = numpy.sum((spectral_train_y-spatial_train_y) ** 2)
        validset_err = numpy.sum((spectral_valid_y-spatial_valid_y) ** 2)
        testset_err  = numpy.sum((spectral_test_y-spatial_test_y) ** 2)
        if testset_err + validset_err + trainset_err == 0:
            print 'Checking test PASSED.'
        else:
            print 'Checking test FAILED.'

    if raw_input('Do you want to save results to data.mat (Y/n)? ') == 'Y':
        print '... saving datasets'
        sio.savemat('data.mat', {'spectral_train_x': dataset_spectral[0][0].get_value(),
                                 'spectral_train_y': dataset_spectral[0][1].get_value(),
                                 'spectral_valid_x': dataset_spectral[1][0].get_value(),
                                 'spectral_valid_y': dataset_spectral[1][1].get_value(),
                                 'spectral_test_x':  dataset_spectral[2][0].get_value(),
                                 'spectral_test_y':  dataset_spectral[2][1].get_value(),
                                 
                                 'spatial_train_x':  dataset_spatial[0][0].get_value(),
                                 'spatial_train_y':  dataset_spatial[0][1].get_value(),
                                 'spatial_valid_x':  dataset_spatial[1][0].get_value(),
                                 'spatial_valid_y':  dataset_spatial[1][1].get_value(),
                                 'spatial_test_x':   dataset_spatial[2][0].get_value(),
                                 'spatial_test_y':   dataset_spatial[2][1].get_value(),
                                 
                                 'extracted_pixel_ind': extracted_pixel_ind,
                                 'split_mask':       split_mask})

        
    print 'Done.'






    '''
    *****old implementation of the part "split dataset"*****
    
    random_mask = rand_num_generator.random_integers(1, sum(ratio), data[0].shape[0])
    split_mask = numpy.array(['tests', ] * data[0].shape[0])
    split_mask[random_mask <= ratio[0]] = 'train'
    split_mask[(random_mask <= ratio[1]+ratio[0]) * (random_mask > ratio[0])] = 'valid'
    
    train_data_x = data[0][split_mask == 'train', :]
    train_data_y = data[1][split_mask == 'train']-1
    valid_data_x = data[0][split_mask == 'valid', :]
    valid_data_y = data[1][split_mask == 'valid']-1
    test_data_x  = data[0][split_mask == 'tests', :]
    test_data_y  = data[1][split_mask == 'tests']-1
    '''
    
