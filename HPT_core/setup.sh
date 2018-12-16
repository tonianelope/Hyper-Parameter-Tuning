#!/bin/bash

pip install -r requierements.txt

mkdir data
mkdir plots
mkdir output

cd data
#wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_images.zip
#unzip mnist_background_images.zip

wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
