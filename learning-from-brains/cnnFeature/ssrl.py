from torch.utils.data.dataloader import DataLoader
import configs
import torch
from torch import nn
import os
from dataset import Dataset
from encDec import encDec
from torch.autograd import Variable
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F
import random
from firstCnnModel import Enc
from secondCnnModel import Dec

if __name__ == '__main__':
    print("Begin ssrlbase")
    if not os.path.exists(configs.resultPath):
        os.makedirs(configs.resultPath)
    torch.manual_seed(configs.seed)
    # model = encDec().cuda()
    # model = model.to(configs.device)
    enc=Enc().cuda()
    dec=Dec().cuda()
    trainDataset = Dataset(configs.dataPath)
    train_size = int(0.8 * len(trainDataset))
    print(train_size)
    val_size = len(trainDataset) - train_size
    print(val_size)
    if configs.validationDataPath==None:
        trainDataset, validationDataset = torch.utils.data.random_split(trainDataset, [train_size, val_size])
    else:
        validationDataset = Dataset(configs.validationDataPath)

    #print(trainDataset)
    trainDataLoader = DataLoader(dataset=trainDataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.threads, pin_memory=True, drop_last=True)
    validationDataLoader = DataLoader(dataset=validationDataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.threads, pin_memory=True, drop_last=True)

    criterion = nn.MSELoss()
    encOptimizer = torch.optim.Adam(enc.parameters(), lr=configs.learning_rate, weight_decay=1e-5)
    decOptimizer = torch.optim.Adam(dec.parameters(), lr=configs.learning_rate, weight_decay=1e-5)

    # adapt these
    for epoch in range(configs.num_epochs):
        with tqdm(trainDataLoader) as tepoch:
            saveImg=0;
            debugImg=0;
            total_loss = 0
            print(tepoch)
            for data in tepoch:
                # print("data.shape")
                # print(data.shape)
                epoch_loss=0
                for timePoint in range(0,data.shape[2]):
                    tepoch.set_description(f"Epoch {epoch} Timepoint {timePoint}")
                    img = data[:,:,timePoint]
                    img = torch.nn.functional.normalize(img,p=2.0,dim=1)
                    maskLength = 1000
                    # print('data.size')
                    # print(data.size())
                    # print('img.size')
                    # print(img.size())
                    rx=random.randrange(0,img.size()[1]-maskLength)
                    #ry=random.randrange(0,300)
                    #cutout= img[:,:,rx:rx+100,ry:ry+100].clone()
                    maskedImg=img.clone()
                    #maskedImg[:,:,rx:rx+100,ry:ry+100]=torch.zeros(1,3,100, 100)
                    maskedImg = Variable(maskedImg).cuda()
                    # print('img.size')
                    # print(img.size())
                    img=Variable(img).cuda()
                    intermediate = enc(maskedImg)
                    # print("intermediate.size")
                    # print(intermediate.size())
                    output=dec(intermediate)
                    # print("output.size")
                    # print(output.size())
                    # lossImg=torch.zeros(20484)
                    # lossImg[:20464]=img
                    lossOut = torch.zeros(40960).cuda()
                    lossOut[:40960] = output
                    loss = criterion(lossOut,img)
                    decOptimizer.zero_grad()
                    loss.backward()
                    decOptimizer.step()
                    total_loss += loss.data
                    epoch_loss += loss.data
                    tepoch.set_postfix(data="train", loss=loss.item(), total_loss=total_loss/trainDataset.__len__(), epoch_loss=epoch_loss /trainDataset.__len__())
                    saveImg=output
            save_image(saveImg, configs.resultPath + '/image_train_{}_{}.png'.format(epoch,epoch_loss))
            # Print model's state_dict
            print("Model's state_dict:")
            for param_tensor in enc.state_dict():
                print(param_tensor, "\t", enc.state_dict()[param_tensor].size())

            # Print optimizer's state_dict
            print("Optimizer's state_dict:")
            for var_name in encOptimizer.state_dict():
                print(var_name, "\t", encOptimizer.state_dict()[var_name])
                # Print model's state_dict

            print("Model's state_dict:")
            for param_tensor in dec.state_dict():
                print(param_tensor, "\t", dec.state_dict()[param_tensor].size())

            # Print optimizer's state_dict
            print("Optimizer's state_dict:")
            for var_name in decOptimizer.state_dict():
                print(var_name, "\t", decOptimizer.state_dict()[var_name])

            torch.save({'epoch': epoch,
            'model_state_dict': enc.state_dict(),
            'optimizer_state_dict': encOptimizer.state_dict(),
            'loss': epoch_loss}, configs.resultPath+"/enc_epoch_{}_loss_{}".format(epoch,epoch_loss))

            torch.save({'epoch': epoch,
                        'model_state_dict': dec.state_dict(),
                        'optimizer_state_dict': decOptimizer.state_dict(),
                        'loss': epoch_loss}, configs.resultPath+"/dec_epoch_{}_loss_{}".format(epoch,epoch_loss))
            #torch.save(dec.state_dict(), configs.resultPath)

            # model = TheModelClass(*args, **kwargs)
            # model.load_state_dict(torch.load(PATH))
            # model.eval()
        # torch.cuda.empty_cache()
        # valLoss=0
        # with tqdm(validationDataLoader) as tepoch:
        #     saveImg=0;
        #     total_loss = 0
        #     for data in tepoch:
        #         tepoch.set_description(f"Epoch {epoch}")
        #
        #         img = data
        #         rx = random.randrange(0, 300)
        #         ry = random.randrange(0, 300)
        #         cutout = img[:, :, rx:rx + 100, ry:ry + 100].clone()
        #         maskedImg = img.clone()
        #         maskedImg[:, :, rx:rx + 100, ry:ry + 100] = torch.zeros(1, 3, 100, 100)
        #
        #         maskedImg = Variable(maskedImg).cuda()
        #         img = Variable(img).cuda()
        #
        #         output = model(maskedImg)
        #         loss = criterion(output, img)
        #
        #         total_loss += loss.data
        #         tepoch.set_postfix(data="val",loss=loss.item(), total_loss=total_loss/validationDataset.__len__())
        #         saveImg=output
        #
        #
        #     save_image(saveImg, configs.resultPath+'/image_val_{}.png'.format(epoch))
        #
        # if epoch % 10 == 0:
        #     torch.save(model.state_dict(), configs.resultPath+'/conv_autoencoder_{}_{}.pth'.format(epoch,valLoss))
