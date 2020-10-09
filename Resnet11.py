import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import sys

from torch.autograd import Variable
import datetime
import os
import numpy as np
import cv2

sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

net = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))
net.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 1)))

# X = torch.rand((1, 3, 224, 224))
# for name, layer in net.named_children():
#     X = layer(X)
#     print(name, ' output shape:\t', X.shape)
# summary(net, (3, 224, 224))
# print("third")

class BatchDataset:
    def __init__(self, path, batchSize = 1 , mode='train'):
        self.img_path = path + '/image_2'
        self.label_path = path + '/label_2'
        self.IDLst = [x.split('.')[0] for x in sorted(os.listdir(self.img_path))]
        self.batchSize = batchSize
        self.mode = mode
        self.imgID = None
        #self.info = self.getBatchInfo()
        #self.Total = len(self.info)
        if mode == 'train':
            self.idx = 0
            self.num_of_patch = 211183
        else:
            self.idx = 0
            self.num_of_patch = 211183

    # def getBatchInfo(self):
    #     buf = []
    #     for index in range(self.batchSize):
    #         buf.append({'ID': self.IDLst[index]})
    #         with open(self.label_path + '/%s.txt'%self.IDLst[index], 'r') as f:
    #             for line in f:
    #                 line = line[:-1].split(' ')
    #                 for i in range(1, len(line)):
    #                     line[i] = float(line[i])
    #                 Class = line[0]
    #                 Location = [line[1], line[2], line[3]] # x, y, z
    #                 Ry = (line[4]) / np.pi * 180    # object yaw
    #                 Dimension = [line[5], line[6], line[7]] # height, width, length
    #                 IoU = line[8]
    #                 top_left = (int(round(line[9])), int(round(line[10])))
    #                 bottom_right = (int(round(line[11])), int(round(line[12])))
    #                 Box_2D = [top_left, bottom_right]
    #                 ThetaRay = (np.arctan2(Location[2], Location[0])) / np.pi * 180
    #
    #                 LocalAngle = 360 - (ThetaRay + Ry)
    #                 if LocalAngle > 360:
    #                     LocalAngle -= 360
    #                 LocalAngle = LocalAngle / 180 * np.pi
    #                 if LocalAngle < 0:
    #                     LocalAngle += 2 * np.pi
    #                 buf.append({
    #                         'Class': Class,
    #                         'Box_2D': Box_2D,
    #                         'Location': Location,
    #                         'Dimension': Dimension,
    #                         'IoU': IoU,
    #                         'Ry': Ry,
    #                         'ThetaRay': ThetaRay,
    #                         'LocalAngle': LocalAngle
    #                     })
    #         img_name = '%s/%s.jpg' % (self.img_path, self.IDLst[index])
    #         img = cv2.imread(img_name, cv2.IMREAD_COLOR).astype(np.float) / 255
    #         img[:, :, 0] = (img[:, :, 0] - 0.406) / 0.225
    #         img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    #         img[:, :, 2] = (img[:, :, 2] - 0.485) / 0.229
    #         buf.append({'Image': img})
    #     return buf

    def getTraindata(self):
        batch = np.zeros([self.batchSize, 3, 224, 224], np.float)
        iou = np.zeros([self.batchSize, 1], np.float)
        buffer = []
        for one in range(self.batchSize):
            with open(self.label_path + '/%s.txt'%self.IDLst[self.idx], 'r') as f:
                for line in f:
                    line = line[:-1].split(' ')
                    for i in range(1, len(line)):
                        line[i] = float(line[i])
                    Class = line[0]
                    IoU = line[8]
                    # IoU = 2*line[8]-1 # map from [0, 1] to [-1, 1]
                    top_left = (int(round(line[9])), int(round(line[10])))
                    bottom_right = (int(round(line[11]+line[9])), int(round(line[12]+line[10])))
                    Box_2D = [top_left, bottom_right]
                    buffer.append({
                            'Class': Class,
                            'Box_2D': Box_2D,
                            'IoU': IoU,
                        })

            buff_data = buffer[one]
            imgID = self.img_path + '/%s.jpg' % self.IDLst[self.idx]
            if imgID != None:
                img = cv2.imread(imgID, cv2.IMREAD_COLOR).astype(np.float) / 255
                img[:, :, 0] = (img[:, :, 0] - 0.406) / 0.225
                img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
                img[:, :, 2] = (img[:, :, 2] - 0.485) / 0.229
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # cv2.namedWindow('GG')
                # cv2.imshow('GG', img)
                # cv2.waitKey(0)

            pt1 = buff_data['Box_2D'][0]
            pt2 = buff_data['Box_2D'][1]
            crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
            crop = cv2.resize(src=crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            batch[one, 0, :, :] = crop[:, :, 2]
            batch[one, 1, :, :] = crop[:, :, 1]
            batch[one, 2, :, :] = crop[:, :, 0]
            iou[one, :] = buff_data['IoU']

            if self.idx + 1 < self.num_of_patch:
                self.idx += 1
            else:
                self.idx = 0

        return batch, iou

    # def getEvaldata(self):
    #     batch = np.zeros([1, 3, 224, 224], np.float)
    #     iou = np.zeros([1, 1], np.float)
    #     for one in range(1):
    #         with open(self.label_path + '/%s.txt'%self.IDLst[self.idx], 'r') as f:
    #             for line in f:
    #                 line = line[:-1].split(' ')
    #                 for i in range(1, len(line)):
    #                     line[i] = float(line[i])
    #                 Class = line[0]
    #                 IoU = line[8]
    #                 top_left = (int(round(line[9])), int(round(line[10])))
    #                 bottom_right = (int(round(line[11])), int(round(line[12])))
    #                 Box_2D = [top_left, bottom_right]
    #                 buffer.append({
    #                         'Class': Class,
    #                         'Box_2D': Box_2D,
    #                         'IoU': IoU,
    #                     })
    #
    #         buff_data = buffer[one]
    #         imgID = self.img_path + '/%s.jpg' % self.IDLst[self.idx]
    #         if imgID != None:
    #             img = cv2.imread(imgID, cv2.IMREAD_COLOR).astype(np.float) / 255
    #             img[:, :, 0] = (img[:, :, 0] - 0.406) / 0.225
    #             img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    #             img[:, :, 2] = (img[:, :, 2] - 0.485) / 0.229
    #             # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #             # cv2.namedWindow('GG')
    #             # cv2.imshow('GG', img)
    #             # cv2.waitKey(0)
    #
    #         pt1 = buff_data['Box_2D'][0]
    #         pt2 = buff_data['Box_2D'][1]
    #         crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
    #         crop = cv2.resize(src=crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    #         batch[one, 0, :, :] = crop[:, :, 2]
    #         batch[one, 1, :, :] = crop[:, :, 1]
    #         batch[one, 2, :, :] = crop[:, :, 0]
    #         iou[one, :] = buff_data['IoU']
    #
    #         if self.idx + 1 < self.num_of_patch:
    #             self.idx += 1
    #         else:
    #             self.idx = 0
    #
    #     return batch, iou

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

if __name__ == '__main__':
    data_path = "/home/bq1235/datasets/cubes"
    store_path = os.path.abspath(os.path.dirname(__file__)) + '/models'
    if not os.path.isdir(store_path):
        os.mkdir(store_path)
    model_lst = [x for x in sorted(os.listdir(store_path)) if x.endswith('.pkl')]

    mode = 'train'
    if mode == "train":
        if len(model_lst) == 0:
            print ('No previous model found, start training')
            net = net
            net.train()
            # model = net.Model(features=vgg.features, bins=bins).cuda()
        else:
            print ('Find previous model %s'%model_lst[-1])
            # model = Model.Model(features=vgg.features, bins=bins)
            # model = Model.Model(features=vgg.features, bins=bins).cuda()
            params = torch.load(store_path + '/%s'%model_lst[-1])
            net.load_state_dict(params)
            net.train()

        loss_stream_file = open('train_data/Loss_record.txt', 'w')
        lr, num_epochs, batch_size = 0.001, 5, 256
        data = BatchDataset(data_path + '/training', batch_size, mode='train')

        net = net.to(device)
        print("training on ", device)
        # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
        iter_each_time = round(float(data.num_of_patch) / batch_size)
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
            for i in range(int(iter_each_time)):           
                batch, iouGT = data.getTraindata()
                batch = Variable(torch.FloatTensor(batch), requires_grad=False).to(device)
                iouGT = Variable(torch.FloatTensor(iouGT), requires_grad=False).to(device)
                iou = net(batch)
                iou_LossFunc = torch.nn.SmoothL1Loss().cuda()
                # iou_LossFunc = torch.nn.MSELoss()
                iou_loss = iou_LossFunc(iou, iouGT)

                optimizer.zero_grad()
                iou_loss.backward()
                optimizer.step()

                train_l_sum += iou_loss.cpu().item()
                train_acc_sum += (iou.argmax(dim=1) == iouGT).sum().cpu().item()
                n += iouGT.shape[0]
                batch_count += 1

                iou = iou.cpu().data.numpy()[0, :]
                iouGT = iouGT.cpu().data.numpy()[0, :]
                loss_stream_file.write('%lf %lf %lf %lf \n ' %(i, iou_loss, iou, iouGT))

                if i % 100 == 0:
                    now = datetime.datetime.now()
                    now_s = now.strftime('%Y-%m-%d-%H-%M-%S')
                    print (' %s Epoch %.2d, IoU Loss: %lf'%(now_s, i, iou_loss))

            # test_acc = evaluate_accuracy(test_iter, net)
            test_acc = 0
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                  % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
            now = datetime.datetime.now()
            now_s = now.strftime('%Y-%m-%d-%H-%M-%S')
            name = store_path + '/model_%s.pkl' % now_s
            torch.save(net.state_dict(), name)
        loss_stream_file.close()

    # else:
        model_lst = [x for x in sorted(os.listdir(store_path)) if x.endswith('.pkl')]
        if len(model_lst) == 0:
            print ('No previous model found, please check it')
            exit()
        else:
            print ('Find previous model %s'%model_lst[-1])
            params = torch.load(store_path + '/%s'%model_lst[-1])
            net.load_state_dict(params)
            net.eval()

        net = net.to(device)
        print("evaluating on ", device)
        iou_stream_file = open('train_data/DNN_result.txt', 'w')
        iou_error = []
        data = BatchDataset(data_path + '/training', batchSize=1, mode='eval')
        with torch.no_grad():
            for i in range(data.num_of_patch):
                batch, iouGT = data.getTraindata()
                batch = Variable(torch.FloatTensor(batch), requires_grad=False).to(device)
                # batch = Variable(torch.FloatTensor(batch), requires_grad=False).cuda()

                iou = net(batch)
                iou = iou.cpu().data.numpy()
                # iou_err = np.mean(abs(np.array(iouGT) - iou))
                iou_err = np.mean(np.array(iouGT) - iou)
                iou_error.append(iou_err)

                # print ('frame: %lf Iou error: %lf %lf %lf '%(i, iou_err, iou, iouGT))
                iou_stream_file.write('%lf %lf %lf %lf\n '%(i,iou_err,  iou, iouGT))

                if i % 1000 == 0:
                    now = datetime.datetime.now()
                    now_s = now.strftime('%Y-%m-%d-%H-%M-%S')
                    print('------- %s %.5d -------' % (now_s, i))
                    print('IoU error: %lf' % (np.mean(iou_error)))
                    print('-----------------------------')

        iou_stream_file.close()






