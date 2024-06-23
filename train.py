"""
    此文件用于模型的训练,训练样例为./datasets/train中的数据，所以在运行此文件之前，请确保./datasets/train
    文件夹下有数据，若没有请用common.py生成数据
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import my_datasets
from model import vkmodel

if __name__ == '__main__':
    train_datas = my_datasets.mydatasets("./datasets/train/")
    train_dataloader = DataLoader(train_datas, batch_size=40, shuffle=True)
    writer = SummaryWriter("logs")

    # 如果支持gpu训练，将开启gpu训练模式
    if torch.cuda.is_available():
        vkmodel = vkmodel().cuda()
        loss_fn = nn.MultiLabelSoftMarginLoss().cuda()
    else:
        vkmodel = vkmodel()
        loss_fn = nn.MultiLabelSoftMarginLoss()

    optimizer = torch.optim.Adam(vkmodel.parameters(), lr=0.001)  # 优化器
    total_step = 0
    # 控制训练层数
    for epoch in range(2):
        print("外层训练次数：{}".format(epoch))
        for i, (images, labels) in enumerate(train_dataloader):
            # 如果支持gpu训练，将开启gpu训练模式
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            vkmodel.train()
            outputs = vkmodel(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_step += 1
            if i % 100 == 0:
                print("训练{}次,损失率:{}".format(i, loss.item()))
                writer.add_scalar("loss", loss, total_step)

torch.save(vkmodel, "model.pth")
