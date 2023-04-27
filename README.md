# pytorchTest
A basic toy example project of using pytorch in c++, and using c++ to extend pytorch operations

## Requires G++11

For ubuntu older than 21.10, run following first

```shell
sudo add-apt-repository 'deb http://mirrors.kernel.org/ubuntu jammy main universe'
sudo apt-get update
```

Then, install the default gcc/g++ of ubuntu22.04

```shell
sudo apt-get install gcc g++ cmake python3 python3-pip
```
## What is this brach?
Use python to create a *.pt which define a VAE, and then use c++ to call it
### First, go into python and run VAE.py
### Second, build the c++ program, and copy the VAE.pt to its build path
### Then we can run the c++ program which automatically builds the VAE according to *.pt
### On going:
- Training in C++
- Make it works in GPU
## Requires Torch

You may refer to https://pytorch.org/get-started/locally/ for mor details, following are the minimal requirements

### (Optional) Cuda-based torch

You may wish to install cuda for faster pre-training on models, following is a reference procedure. Please refer
to https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu

```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda
sudo apt-get install nvidia-gds
sudo apt-get install libcudnn8 libcudnn8-dev libcublas-11-7
```
The libcublas depends on your specific version.
Then you may have to reboot for enabling cuda.

### (Required) Install pytorch

```shell
sudo apt-get install python3 python3-pip
```

(w/ CUDA):
(Please make all cuda dependencies installed before pytorck!!!)
```shell
pip3 install torch torchvision torchaudio
```

(w/o CUDA)

```shell
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```
### (Required) Install graphviz
```shell
sudo apt-get install graphviz
pip install torchviz
```
## How to build
(CUDA-related is only necessary if your pytorch has cuda, but it's harmless if you don't have cuda.)
### Build in shell

```shell
export CUDACXX=/usr/local/cuda/bin/nvcc
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
make 
```

### Tips for build in Clion
There are bugs in the built-in cmake of Clion, so you can not run
-DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`.
Following may help:

- Please run 'import torch;print(torch.utils.cmake_prefix_path)' manually first, and copy the path
- Paste the path to -DCMAKE_PREFIX_PATH=
- Manually set the environment variable CUDACXX as "/usr/local/cuda/bin/nvcc" in Clion's cmake settings

## Test
- run pytorch in c++: ./example-app
- run our own extension in python python3 pythonTest.py

## Local generation of the documents

You can also re-generate them locally, if you have the doxygen and graphviz. Following are how to install them in ubuntu
21.10/22.04

```shell
sudo apt-get install doxygen
sudo apt-get install graphviz
```

Then, you can do

```shell
mkdir doc
doxygen Doxyfile
```

to get the documents in doc/html folder, and start at index.html


## Known issues

1. If you use Torch with cuda, the nvcc will refuse to work as it doesn't support c++20 yet. Therefore, we disabled the
   global requirement check of C++ 20, and only leave an "-std=c++20" option for g++. This will be fixed after nvidia
   can support c++20 in cuda.
