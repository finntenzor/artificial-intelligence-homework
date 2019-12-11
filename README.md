# Artificial Intelligence Homework

## 综述

基于CUDA & C++的CNN实现，在MNIST上（也不支持别的数据集了—）大约达到98%的正确率，需要NVIDIA显卡支持。

## 编译

需要CUDA编译套件，仅在Windows10+CUDA10.0+VisualStudio2017上测试过能正常编译。

附：Cuda下载和安装

[CUDA10.0官网下载](https://developer.nvidia.com/cuda-10.0-download-archive)

[CUDA10.0与VS冲突时的解决方案](https://blog.csdn.net/zzpong/article/details/80282814)

注：你可能遇到别的问题，届时请自行百度/Google

编译时请检查本机包括make和nvcc：

```sh
$ nvcc -V
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2018 NVIDIA Corporation
# Built on Sat_Aug_25_21:08:04_Central_Daylight_Time_2018
# Cuda compilation tools, release 10.0, V10.0.130
```

```sh
$ make -V
# D:\Programs\mingw64\bin\make: invalid option -- V
# Usage: make [options] [target] ...
```

也可以不使用make进行编译，若不使用make，你需要对src下每一个cpp和cu文件单独编译，类似：

```sh
$ nvcc -c ./src/cli.cpp -o ./obj/cli.obj -rdc=true -I ./include/
$ nvcc -c ./src/config.cpp -o ./obj/config.obj -rdc=true -I ./include/
$ nvcc -c ./src/layer_core_convolution.cu -o ./obj/layer_core_convolution.obj -rdc=true -I ./include/
# ...
$ nvcc ./obj/cli.obj ./obj/config.obj ./obj/layer_core_convolution.obj ... -o main.exe
```

你可能注意到根目录有一个python文件，这个文件仅用于将src中几个文件的后缀名在cu和cpp之间转换，编译本身不需要python，这个转换仅仅是为了在编写代码时能让我的VSCode开启代码提示，请注意以下文件后缀名必须是cu。

```txt
layer_core_convolution.cu
layer_core_dense.cu
layer_core_input.cu
layer_core_output.cu
layer_core_pooling.cu
layer_core_relu.cu
layer_core_tanh.cu
layer_core_scale.cu
layer_run_common.cu
model_run.cu
```

## 运行

### 训练

```sh
./main.exe --train --config=mnist.config
```

或

```sh
./main.exe -t -c=mnist.config
```

### 测试

```sh
./main.exe --predict --config=mnist.config
```

或

```sh
./main.exe -p -c=mnist.config
```

### 其他

程序可以接受的其他命令行参数如：

* --gpu 或 -g 显示当前机器上所有的GPU和其相关参数
* --device=x 或 -d=x 使用第x个显卡进行训练或预测
* --train 或 -t 进行训练
* --predict 或 -p 进行测试
* --load ./xxx 或 -l ./xxx 将当前目录下的文件xxx作为模型读入
* --save ./xxx 或 -s ./xxx 将模型保存至当前目录下的xxx（需要训练时才能保存模型）
* --config ./xxx 或 -c ./xxx 将当前目录下的文件xxx作为配置文件设置参数

其中load、save和config参数都需要一个参数，可以用等号也可以用空格，即`--config=xxx`和`--config xxx`是等效的

### 配置文件

配置文件大致如下：

```ini
[input]
trainImage ./data/train-images.idx3-ubyte # 训练集x路径
trainLabel ./data/train-labels.idx1-ubyte # 训练集y路径
testImage ./data/t10k-images.idx3-ubyte # 测试集x路径
testLabel ./data/t10k-labels.idx1-ubyte # 测试集y路径
[global]
loadPath ./mnist.5.model # 模型加载路径
savePath ./mnist.5.model # 模型保存路径
epoch 10 # 将整个训练集运行多少遍
batchSize 100 # 每一批训练多少张图片
trainImageCount 60000 # 训练集图片个数
testImageCount 10000 # 测试集图片个数
predictImageCount 10000 # 使用测试集中多少个图片进行预测
predictOutputCount 20 # 输出前多少个测试集图片的预测结果
studyRate 0.001 # 学习率 由于使用Adam算法请不要调整
attenuationRate 1 # 衰减率 由于使用Adam算法请不要调整
printMemoryUsed 1 # 是否输出显存占用
printTrainProcess 1 # 是否在训练过程中输出当前训练结果
printPredictOutput 1 # 是否输出在测试集上的预测结果
printPredictAccuracyRate 1 # 是否输出在测试集上的正确率
printModelSchema 0 # 是否输出模型的各层输入、输出形状
lossCheckCount 3 # 检查多少个loss来判断是否提前退出，不少于2
[model] # 模型，即各层参数，可调
Input 28 28 # 输入层，必须是第一层，接受两个参数宽 高
Convolution 20 5 1 0 # 卷积层 参数依次为 输出通道数 卷积核大小 步长 步长偏置=0 输出大小=自动
Pooling 2 # 池化层 参数依次为 池化窗口大小 步长=自动（窗口大小） 步长偏置=0 输出大小=自动
Convolution 50 5 1 0
Pooling 2
Dense 500 # 全连接层 参数为隐层神经元个数
Relu # 激活函数层 无参数，目前仅支持Relu
Dense 10 # 由于输出10个参数，因此最后一层必须是Dense 10
Output # 输出层，必须是最后一层，使用softmax激活，目前没得选择
```

注：

* 若命令行参数已经给出loadPath或savePath，则命令行参数优先
* 卷积、池化的参数可以是非方形的，具体间源代码（model_facade_builder.h）中的定义（似乎没什么用）

## 源码

### 目录结构

```txt
--
 |- data mnist数据集（为了方便数据集已经传上git，不需要另外下载）
 |- include 头文件
 |- obj 编译目标文件
 |- src 源代码
 |- Makefile
 |- mnist.config 编译配置
 |- README.md 本文件
```

### 主要代码

* 各层的正向和反向运算 `/src/layer_core_xxx.cu` 其中xxx是层的类型。（注：scale已经废弃）
* 模型的正向和反向运算流程、下降算法 `/src/model_run.cu`
* 入口 `/src/main.cpp`
* 其他代码
  * 命令行参数解析 `/src/cli.cpp`
  * 配置文件解析 `/src/config.cpp`
  * 训练集、测试集文件读取 `/src/read_file.cpp`
  * 中间变量输出（调试用） `/src/visual_funcs.cpp`和`/src/main_debug.cpp`
* 显存分配
  * `/src/layer_memory.cpp`
  * `/src/model_memory.cpp`
* Facade
  * 层 `/src/layer_facade.cpp`，用于将某一层的参数从显存拷贝回内存，用于debug
  * 模型 `/src/model_facade.cpp` 用于调用`/src/model_run.cu`和一些其他模型中的函数
  * builder `/src/model_facade_builder.cpp` 用于提供构建模型的API构建

### 其他源码注意事项

* 由于作者太菜不会调整nvcc的编码，因此所有源代码使用GBK编码而不是utf-8编码
* 注意头文件和代码文件并不是一一对应的，主要是各层的正向和反向的代码并不是和头文件一一对应的
* 如果需要修改源码，你可能需要学习基本的CUDA编程，了解基于GPU的多线程编程基本方法
* 代码性能很差，但是作者太菜不会优化
* 权重初始化在`/src/model_facade.cpp`中，下降算法在`/src/model_run.cu`中，他们的正确与否非常影响模型能否训练
* 代码中所有的层的输入和输出均为4维张量，但是均以一维数组保存，这导致了下标运算非常容易出错并且难以调试（但是节约了显存，并且简化了下降算法的代码），在编写各层的正向反向传播算法时请特别注意下标运算，下标的错误非常难以调试。

## 写在最后

Under [MIT License](LICENSE)

求Star
