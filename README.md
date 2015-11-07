# Deep Learning on Hyperspectral Image Classification

This repository guarantees you to reproduce the results reported in:
 - [Spectral-spatial classification of hyperspectral image using autoencoders](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=6782778&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel7%2F6777690%2F6782769%2F06782778.pdf%3Farnumber%3D6782778)
 - [Deep Learning-Based Classification of Hyperspectral Data](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6844831)

## Setup
### Install Dependencies
 - Theano
If you were using Ubuntu, simply type 

    sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
    sudo pip install Theano

in your terminal. If you were not in Ubuntu, you can find all the information you need at [here](http://deeplearning.net/software/theano/).

 - scikit-learn
Scikit-learn is used to run control group experiments, you can skip it if you don't care about reproducing those SVM results or confusion matrices. To install it, simply type the following command:

    pip install -U scikit-learn

 - PIL
PIL is used to visualize weights and classification results on the whole image. It could also be skipped. To install it, type:

    sudo apt-get install libjpeg libjpeg-dev libfreetype6 libfreetype6-dev zlib1g-dev
    pip install PIL

 - CUDA
If you have a CUDA enabled GPU with you, that would save you a lot of time. But it is still OK to run all the experiments without GPU, at the expense of more patience.


### Download ang prepare the datasets
Download the two datasets used in the paper:
 - [Pavia University scene](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)
 - [Kennedy Space Center (KSC)](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)

Change the loading directory in each .py file to wherever you store your datasets.

## Reproducing the results

1. To train a stacked autoencoder on spectral information, execute

    python pavia_SdA.py

   for Pavia dataset, or 

    python ksc_SdA.py

   for KSC dataset.

   These experiments correspond to the classical way of classifying spectra,
   without considering the spatial correlations in HSI.

2. To train the model on spatial information, execute

    python pavia_spatial_SdA.py

   for Pavia dataset, or 

    python ksc_spatial_SdA.py

   for KSC dataset.

   These experiments extract features in a novel way. Those features emphasize on spatial correlations in an HSI dataset. It shows that even discarding the majority of spectral dimensions, classification could still be performed successfully.

3. To train the model on joint spectral-spatial information, execute

    python pavia_joint_SdA.py

   for Pavia dataset, or 

    python ksc_joint_SdA.py

   for KSC dataset.

   These experiments utilizes both spectral and spatial information. They reach the state-of-the-art on both datasets, surpassing a SVM with heavily tuned parameters. 

## Contact:
Zhouhan Lin: zhouhan.lin [at] umontreal.ca
