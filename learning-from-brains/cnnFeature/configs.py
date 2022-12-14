import torch
#dataPath="C:/Users/ghait/Repos/SSFLData/train"
dataPath="C:/Users/ghait/Repos/learningFromBrains/hcp/100307/analysis"
#validationDataPath="C:/Users/ghait/Repos/SSFLData/val"
validationDataPath=None
resultPath="C:/Users/ghait/Repos/learningFromBrains/learning-from-brains/cnnFeature/results"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate=0.0005
num_epochs=1500
seed=1
batch_size=1
threads=1