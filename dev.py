import os
import shutil
import sys

layers = [
    'core_input',
    'core_convolution',
    'core_pooling',
    'core_dense',
    'core_scale',
    'core_output',
    'run_common',
]

fileNames = ['./src/layer_' + layerName for layerName in layers]
fileNames.append('./src/model_run')

def move(fromSt='.cpp', toSt='.cu'):
    for fileName in fileNames:
        fromName = fileName + fromSt
        toName = fileName + toSt
        if os.path.exists(fromName):
            shutil.move(fromName, toName)

def main():
    mode = sys.argv[1] if len(sys.argv) >= 2 else None
    if mode == 'dev':
        move('.cu', '.cpp')
    elif mode == 'prod':
        move('.cpp', '.cu')
    else:
        print("Unkown mode " + str(mode))

if __name__ == '__main__':
    main()
