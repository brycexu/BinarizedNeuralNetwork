This repository contains my evaluations of several Binarized Neural Networks inspired by the idea in:

https://arxiv.org/abs/1602.02830

Copyright: Xianda Xu xiandaxu@std.uestc.edu.cn 

LeNet (Tensorflow)
------
Dataset: MNIST

Netscope: [Network](http://ethereon.github.io/netscope/#/gist/8adf14a931afaba55cd3879c80a1c710)

Baseline Accuracy: 99.15%

Binarized Accuracy: 92.87%

Adapted AlexNet (PyTorch)
------
Dataset: CIFAR-10

Netscope: [Network](http://ethereon.github.io/netscope/#/gist/20cb5ef30b0c43a2f33c5d9625354b16)

Baseline Accuracy: 90.78%

Binarized Accuracy: 85.66%

<div align=center><img width="453" height="200" src="https://github.com/brycexu/BinarizedNeuralNetwork/blob/master/Adapted%20AlexNet/Images/Binarized.png"/></div>

Quantized Accuracy: 85.52% (8-bit training: https://arxiv.org/abs/1805.11046)

<div align=center><img width="453" height="200" src="https://github.com/brycexu/BinarizedNeuralNetwork/blob/master/Adapted%20AlexNet/Images/Quantized.png"/></div>

Vgg Net (PyTorch)
------
Dataset: CIFAR-10

Netscope: [Network](http://ethereon.github.io/netscope/#/gist/580c09931dcf801a79dacf7bb6ea7e3a)

Baseline Accuracy: 90.93%

Binarized Accuracy: 81.16%

<div align=center><img width="453" height="200" src="https://github.com/brycexu/BinarizedNeuralNetwork/blob/master/VggNet/Images/Vgg.png"/></div>
