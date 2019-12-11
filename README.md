# Artificial Intelligence Homework

## ����

����CUDA & C++��CNNʵ�֣���MNIST�ϣ�Ҳ��֧�ֱ�����ݼ��ˡ�����Լ�ﵽ98%����ȷ�ʣ���ҪNVIDIA�Կ�֧�֡�

## ����

��ҪCUDA�����׼�������Windows10+CUDA10.0+VisualStudio2017�ϲ��Թ����������롣

����Cuda���غͰ�װ

[CUDA10.0��������](https://developer.nvidia.com/cuda-10.0-download-archive)

[CUDA10.0��VS��ͻʱ�Ľ������](https://blog.csdn.net/zzpong/article/details/80282814)

ע�����������������⣬��ʱ�����аٶ�/Google

����ʱ���鱾������make��nvcc��

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

Ҳ���Բ�ʹ��make���б��룬����ʹ��make������Ҫ��src��ÿһ��cpp��cu�ļ��������룬���ƣ�

```sh
$ nvcc -c ./src/cli.cpp -o ./obj/cli.obj -rdc=true -I ./include/
$ nvcc -c ./src/config.cpp -o ./obj/config.obj -rdc=true -I ./include/
$ nvcc -c ./src/layer_core_convolution.cu -o ./obj/layer_core_convolution.obj -rdc=true -I ./include/
# ...
$ nvcc ./obj/cli.obj ./obj/config.obj ./obj/layer_core_convolution.obj ... -o main.exe
```

�����ע�⵽��Ŀ¼��һ��python�ļ�������ļ������ڽ�src�м����ļ��ĺ�׺����cu��cpp֮��ת�������뱾����Ҫpython�����ת��������Ϊ���ڱ�д����ʱ�����ҵ�VSCode����������ʾ����ע�������ļ���׺��������cu��

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

## ����

### ѵ��

```sh
./main.exe --train --config=mnist.config
```

��

```sh
./main.exe -t -c=mnist.config
```

### ����

```sh
./main.exe --predict --config=mnist.config
```

��

```sh
./main.exe -p -c=mnist.config
```

### ����

������Խ��ܵ����������в����磺

* --gpu �� -g ��ʾ��ǰ���������е�GPU������ز���
* --device=x �� -d=x ʹ�õ�x���Կ�����ѵ����Ԥ��
* --train �� -t ����ѵ��
* --predict �� -p ���в���
* --load ./xxx �� -l ./xxx ����ǰĿ¼�µ��ļ�xxx��Ϊģ�Ͷ���
* --save ./xxx �� -s ./xxx ��ģ�ͱ�������ǰĿ¼�µ�xxx����Ҫѵ��ʱ���ܱ���ģ�ͣ�
* --config ./xxx �� -c ./xxx ����ǰĿ¼�µ��ļ�xxx��Ϊ�����ļ����ò���

����load��save��config��������Ҫһ�������������õȺ�Ҳ�����ÿո񣬼�`--config=xxx`��`--config xxx`�ǵ�Ч��

### �����ļ�

�����ļ��������£�

```ini
[input]
trainImage ./data/train-images.idx3-ubyte # ѵ����x·��
trainLabel ./data/train-labels.idx1-ubyte # ѵ����y·��
testImage ./data/t10k-images.idx3-ubyte # ���Լ�x·��
testLabel ./data/t10k-labels.idx1-ubyte # ���Լ�y·��
[global]
loadPath ./mnist.5.model # ģ�ͼ���·��
savePath ./mnist.5.model # ģ�ͱ���·��
epoch 10 # ������ѵ�������ж��ٱ�
batchSize 100 # ÿһ��ѵ��������ͼƬ
trainImageCount 60000 # ѵ����ͼƬ����
testImageCount 10000 # ���Լ�ͼƬ����
predictImageCount 10000 # ʹ�ò��Լ��ж��ٸ�ͼƬ����Ԥ��
predictOutputCount 20 # ���ǰ���ٸ����Լ�ͼƬ��Ԥ����
studyRate 0.001 # ѧϰ�� ����ʹ��Adam�㷨�벻Ҫ����
attenuationRate 1 # ˥���� ����ʹ��Adam�㷨�벻Ҫ����
printMemoryUsed 1 # �Ƿ�����Դ�ռ��
printTrainProcess 1 # �Ƿ���ѵ�������������ǰѵ�����
printPredictOutput 1 # �Ƿ�����ڲ��Լ��ϵ�Ԥ����
printPredictAccuracyRate 1 # �Ƿ�����ڲ��Լ��ϵ���ȷ��
printModelSchema 0 # �Ƿ����ģ�͵ĸ������롢�����״
lossCheckCount 3 # �����ٸ�loss���ж��Ƿ���ǰ�˳���������2
[model] # ģ�ͣ�������������ɵ�
Input 28 28 # ����㣬�����ǵ�һ�㣬�������������� ��
Convolution 20 5 1 0 # ����� ��������Ϊ ���ͨ���� ����˴�С ���� ����ƫ��=0 �����С=�Զ�
Pooling 2 # �ػ��� ��������Ϊ �ػ����ڴ�С ����=�Զ������ڴ�С�� ����ƫ��=0 �����С=�Զ�
Convolution 50 5 1 0
Pooling 2
Dense 500 # ȫ���Ӳ� ����Ϊ������Ԫ����
Relu # ������� �޲�����Ŀǰ��֧��Relu
Dense 10 # �������10��������������һ�������Dense 10
Output # ����㣬���������һ�㣬ʹ��softmax���Ŀǰû��ѡ��
```

ע��

* �������в����Ѿ�����loadPath��savePath���������в�������
* ������ػ��Ĳ��������ǷǷ��εģ������Դ���루model_facade_builder.h���еĶ��壨�ƺ�ûʲô�ã�

## Դ��

### Ŀ¼�ṹ

```txt
--
 |- data mnist���ݼ���Ϊ�˷������ݼ��Ѿ�����git������Ҫ�������أ�
 |- include ͷ�ļ�
 |- obj ����Ŀ���ļ�
 |- src Դ����
 |- Makefile
 |- mnist.config ��������
 |- README.md ���ļ�
```

### ��Ҫ����

* ���������ͷ������� `/src/layer_core_xxx.cu` ����xxx�ǲ�����͡���ע��scale�Ѿ�������
* ģ�͵�����ͷ����������̡��½��㷨 `/src/model_run.cu`
* ��� `/src/main.cpp`
* ��������
  * �����в������� `/src/cli.cpp`
  * �����ļ����� `/src/config.cpp`
  * ѵ���������Լ��ļ���ȡ `/src/read_file.cpp`
  * �м��������������ã� `/src/visual_funcs.cpp`��`/src/main_debug.cpp`
* �Դ����
  * `/src/layer_memory.cpp`
  * `/src/model_memory.cpp`
* Facade
  * �� `/src/layer_facade.cpp`�����ڽ�ĳһ��Ĳ������Դ濽�����ڴ棬����debug
  * ģ�� `/src/model_facade.cpp` ���ڵ���`/src/model_run.cu`��һЩ����ģ���еĺ���
  * builder `/src/model_facade_builder.cpp` �����ṩ����ģ�͵�API����

### ����Դ��ע������

* ��������̫�˲������nvcc�ı��룬�������Դ����ʹ��GBK���������utf-8����
* ע��ͷ�ļ��ʹ����ļ�������һһ��Ӧ�ģ���Ҫ�Ǹ��������ͷ���Ĵ��벢���Ǻ�ͷ�ļ�һһ��Ӧ��
* �����Ҫ�޸�Դ�룬�������Ҫѧϰ������CUDA��̣��˽����GPU�Ķ��̱߳�̻�������
* �������ܺܲ��������̫�˲����Ż�
* Ȩ�س�ʼ����`/src/model_facade.cpp`�У��½��㷨��`/src/model_run.cu`�У����ǵ���ȷ���ǳ�Ӱ��ģ���ܷ�ѵ��
* ���������еĲ������������Ϊ4ά���������Ǿ���һά���鱣�棬�⵼�����±�����ǳ����׳��������Ե��ԣ����ǽ�Լ���Դ棬���Ҽ����½��㷨�Ĵ��룩���ڱ�д����������򴫲��㷨ʱ���ر�ע���±����㣬�±�Ĵ���ǳ����Ե��ԡ�

## д�����

Under [MIT License](LICENSE)

��Star
