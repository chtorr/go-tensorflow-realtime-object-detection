# go-tensorflow-object-detection
Real-time object detection with Go, Tensorflow, and OpenCV

## Overview

This is a small demo app using Go, Tensorflow, and OpenCV to detect objects objects in real time using the Google provided Tensorflow object detection models. 

* The models are part of the Tensorflow Objecte Dection API project: https://github.com/tensorflow/models/tree/master/research/object_detection.
* The prediction code is laregely adapted from https://github.com/ActiveState/gococo.
* The models are available at https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md.
* This project uses the awesome Pixel to display things, 

On a 2015 Macbook Pro using the SSD mobilenet model and CPU-only inference I can get ~8 FPS, compiling Tensorflow from source with AVX2 support I can get ~11 FPS, and customizing the stock models to have a high score_threshold gets ~14 FPS.

## Installation

1. Git clone this repo
2. Install OpenCV following these instruction: https://github.com/hybridgroup/gocv#how-to-install
   - the gocv package is included in this repo's vendor folder, you should only need to complete the OpenCV installation instructions
3. Install Tensorflow https://www.tensorflow.org/install/install_go
   - You should be able to install Tensorflow for GPU inference without making any changes to this demo
   - If you want to build from source follow this doc https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/README.md
     - I used this command to build with extra CPU features : `bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 -k //tensorflow:libtensorflow.so`
4. Download and extract a model from the model zoo https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
5. `go run src/main.go -device 0 -model <path_to_model_dir>`
