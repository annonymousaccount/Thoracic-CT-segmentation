# Thoracic-CT-segmentation
This is a repository containing the main framework for our submitted work to MICCAI 2017 (Joint Segmentation of Multiple Thoracic Organs in CT Images with Two Collaborative Deep Architectures)

You will need [Caffe](https://github.com/BVLC/caffe),Python,[SimpleITK](https://itk.org/Wiki/SimpleITK/GettingStarted).
For Caffe you would need the [CRFasRNN](https://github.com/torrvision/crfasrnn) layers.

This is the main framweork:

![alt text](https://writelatex.s3.amazonaws.com/psrrhdssfzgp/uploads/4427/10251954/1.png "framework")

It consists of basically two processes: 1st level and 2nd level.
The first level is a FCN+CRFasRNN whose model is the folder 1st_levelcrf.
To train this model, you would first need to train the network without the CRF module in order to obtain a good initialization of the probability maps. Once you have a somewhat good model, you can insert the CRFasRNN module and train a new model.
The second step will use the outputs of the 1st level module in order to train the second level networks. This model is in the folder 2nd_level_exclusion. Each second level will segment an organ individually by using the CT images as input concatenated with the maps of the other organs (excluding the specific organ to segment). After training, you can use the eval scripts provided at each level to generate the segmentation maps on test images and you can obtain segmentation maps like this:

![alt text](https://writelatex.s3.amazonaws.com/psrrhdssfzgp/uploads/1/9467802/47.png)




