import torch
import os
import pathlib
#from dirname import basePath
dataPath="C:/Users/ghait/Repos/SSFLData/train"
dataPath3D="C:/Users/ghait/Repos/SSFLData/train"
#dataPath=basePath+"/hcp/100307/analysis"
basePath=pathlib.Path().absolute().as_posix()
resultPath=basePath+"/results"

if  'lin' in pathlib.Path().absolute().as_posix():
    dataPath="/space_lin1/hcp"
if 'lin' in pathlib.Path().absolute().as_posix():
    dataPath3D = "/space_lin1/hcp/"

#dataPath="C:/Users/ghait/Repos/learningFromBrains/hcp/100307/analysis"
#validationDataPath="C:/Users/ghait/Repos/SSFLData/val"
validationDataPath=None
#resultPath="C:/Users/ghait/Repos/learningFromBrains/learning-from-brains/cnnFeature/results"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate=0.0005
num_epochs=1
seed=1
batch_size=1
threads=1
