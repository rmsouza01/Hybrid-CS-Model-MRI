# Hybrid-CS-Model-MRI


Reconstructing magnetic resonance (MR) images from undersampled k-space measurements can potentially decrease MR examination times. This repository contains thesource code for a hybrid k-space/image domain model proposed by our group. Our model is called W-net. If you use this code in your experiments, we ask you to kindly cite our paper:

Souza, Roberto, and Richard Frayne. "A hybrid frequency-domain/image-domain deep network for magnetic resonance image reconstruction." In 2019 32nd SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), pp. 257-264. IEEE, 2019. (https://ieeexplore.ieee.org/abstract/document/8919674?casa_token=8UZWj9Zs9EoAAAAA:zIhNEBpZ8wkkpksvVNJqV4OQ9HBB-XKJaI5RQ--3GF9xzER00-_LOPEcgMO0vXjbce1utdzuXyI). 

## Dataset

The MR raw-data is publicly available as part of the Calgary-Campinas dataset (https://sites.google.com/view/calgary-campinas-dataset/home) for benhcmarking purposes.


## Code
The code was developed using Python 2.7, NumPy, TensorFlow and Keras. It should be easy to use it, but we appreciate any feedback on how to improve our repository.


## W-net
![w-net architecture](./Figs/w-net.png?raw=True)
W-net architecture. It is composed of a residual U-net on k-space domain connected to an image domain U-net through the magnitude of the inverse discrete fourier transform operator.

![Sample Reconstruction](./Figs/sample_rec.png?raw=True)
Sample reconstruction of our W-net and four other methods published in the literature with special highlight on the cerebellum region. Speed-up factor of 5x.

![w-net reconstruction](./Figs/hybrid_5x.gif?raw=True)

W-net reconstruction gif. From left to right: fully sampled reconstruction, W-net reconstruction from a k-space undersample by 80%, and absolute error differences.

## Contact
Any questions? roberto.medeirosdeso@ucalgary.ca

MIT License
Copyright (c) 2017 Roberto M Souza
