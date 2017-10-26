# Real-time object detection with Go, Tensorflow, and OpenCV

## Overview

This is a small demo app using Go, Tensorflow, and OpenCV to detect objects objects in real time using the Google provided Tensorflow object detection models. 

The models are part of the [Tensorflow Object Dection API](https://github.com/tensorflow/models/tree/master/research/object_detection) project.

The prediction code is laregely adapted from https://github.com/ActiveState/gococo.

The object detection models are available in the [model detection zoo]( https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

This project uses [Pixel](https://github.com/faiface/pixel) to display everything.

![demo gif](https://user-images.githubusercontent.com/3046275/32033304-dbaa1556-b9d9-11e7-8c60-fa9403c31fff.gif)

## Performance
On a 2015 Macbook Pro using the SSD mobilenet model and CPU-only inference I can get ~15-20 fps.  The non-mobilenet models are of course much slower.  Compiling tensorflow with AVX support gave a boost of 2-3 fps.  Additional speed can be gained [tuning the model score_threshold and re-exporting](https://github.com/tensorflow/models/issues/1609#issuecomment-309502384).

## Installation

I've not vendored dependencies as they rely on system level C libraries, which makes things a bit funky.  It's doable, but not worth it for a demo.

1. Git clone this repo
2. Install gocv and OpenCV following these instruction: https://github.com/hybridgroup/gocv#how-to-install
3. Install Tensorflow and the go bindings: https://www.tensorflow.org/install/install_go
   - You should be able to install Tensorflow for GPU inference without making any changes to this demo
   - If you want to build from source follow this doc https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/README.md
     - I used this command to build with extra CPU features : `bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 -k //tensorflow:libtensorflow.so`
4. Install Pixel https://github.com/faiface/pixel#pixel----
4. Download and extract a model from the model zoo https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
5. `go run src/main.go -device 0 -dir <path_to_model_dir>`


You should be able to execute something like this on osx:
```sh
# this repo
git clone https://github.com/chtorr/go-tensorflow-object-detection.git
cd go-tensorflow-objection-detection
# gocv
brew install opencv
go get -u -d gocv.io/x/gocv
source $GOPATH/src/gocv.io/x/gocv/env.sh
# tensorflow
TF_TYPE="cpu" # Change to "gpu" for GPU support
 TARGET_DIRECTORY='/usr/local'
 curl -L \
   "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-$(go env GOOS)-x86_64-1.3.0.tar.gz" |
 sudo tar -C $TARGET_DIRECTORY -xz
go get github.com/tensorflow/tensorflow/tensorflow/go
# pixel
go get github.com/faiface/pixel
# get the ssd mobilenet model
mkdir -p models
curl -L "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz" | tar -C models -xz
# run it
go run src/main.go -device 0 -dir models/ssd_mobilenet_v1_coco_11_06_2017
```
