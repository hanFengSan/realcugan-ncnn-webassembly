# RealCUGAN-ncnn-webassembly
本项目使用Web Assembly技术，基于ncnn运行Real-CUGAN模型。目前只能使用CPU进行计算。
[Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) 是一个使用百万级动漫数据进行训练的，结构与Waifu2x兼容的通用动漫图像超分辨率模型。它支持2x\3x\4x倍超分辨率，其中2倍模型支持4种降噪强度与保守修复，3倍/4倍模型支持2种降噪强度与保守修复。

# How to build
 1. 安装[emscripten](https://github.com/emscripten-core/emscripten)
 ```shell
 git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install 3.1.13
./emsdk activate 3.1.13

source emsdk/emsdk_env.sh # 或者添加到.zshrc等地方
```
2. 构建项目:
```shell
git clone https://github.com/hanfengsan/realcugan-ncnn-webassembly.git
cd realcugan-ncnn-webassembly

git submodule update --init
sh build.sh

go run local_server.go
```
然后打开: http://localhost:8000
