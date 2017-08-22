# SegNets
This is the holding place for experiments with differnt deep learning models for Image segmentation in Keras/Tensorflow. The goal is to create an up-to-date assessment of performance trade-offs when using different state of the art architectures on different platforms (cloud/server/embedded).

## Segnet

Architecture:
Segnet Basic and Segnet Full are inspired by the architectures presented at:
http://arxiv.org/abs/1511.02680
Alex Kendall, Vijay Badrinarayanan and Roberto Cipolla "Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding." arXiv preprint arXiv:1511.02680, 2015.

http://arxiv.org/abs/1511.00561
Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." PAMI, 2017. 

Training Data:
CamVid dataset at:
https://github.com/alexgkendall/SegNet-Tutorial

## U-Net

U-net Models
https://arxiv.org/abs/1505.04597

## VGG-16

This is a reference implementation of the VGG-16 architecture in Keras adapted from the one placed in the Keras repo.

Architecture:
https://arxiv.org/abs/1409.1556
Karen Simonyan, Andrew Zisserman "Very Deep Convolutional Networks for Large-Scale Image Recognition" ICLR, 2015 

Training Data:
ImageNet:
http://www.image-net.org/

## Docker Containerized Workflow

INSTALL docker
BUILD container: $docker build -t segnet-keras .
RUN container: $docker run -it -v local_root_folder:/app segnet-keras /bin/bash

## Coming Soon

Faster RCNNs/Region Proposal Networks

Docker container and script for quick evaluation of reference models

## License
MIT license for research and personal use.

## Contact
@xuberance137

