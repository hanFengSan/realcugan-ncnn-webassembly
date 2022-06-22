# RealCUGAN-ncnn-webassembly
[中文](https://github.com/hanFengSan/realcugan-ncnn-webassembly/blob/main/README_CN.md)

This project uses Web Assembly technology to run the Real-CUGAN model based on ncnn. 

[Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) is an AI super resolution model for anime images, trained in a million scale anime dataset, using the same architecture as Waifu2x-CUNet. It supports 2x\3x\4x super resolving. For different enhancement strength, now 2x Real-CUGAN supports 5 model weights, 3x/4x Real-CUGAN supports 3 model weights.

The code implementation deeply refers to [realcugan-ncnn-vulkan](https://github.com/nihui/realcugan-ncnn-vulkan) and [ncnn-webassembly-nanodet](https://github.com/nihui/ncnn-webassembly-nanodet).

# Usage
Website： https://real-cugan.animesales.xyz/

iOS: is not currently supported.

Android: please open in a browser app.

PC/Mac/Linux: recommend using the latest version of Chrome or Firefox.

# How to build
 1. Install [emscripten](https://github.com/emscripten-core/emscripten):
 ```shell
 git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install 3.1.13
./emsdk activate 3.1.13

source emsdk/emsdk_env.sh # or add it to .zshrc etc.
```
2. build:
```shell
git clone https://github.com/hanfengsan/realcugan-ncnn-webassembly.git
cd realcugan-ncnn-webassembly

git submodule update --init
sh build.sh

go run local_server.go
```
open in brower: http://localhost:8000
