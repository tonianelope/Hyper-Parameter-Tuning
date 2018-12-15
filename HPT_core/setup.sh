#!/bin/bash

pip install -r requierements.txt

mkdir data
mkdir plots
mkdir output

cd data
wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_images.zip
unzip mnist_background_images.zip
