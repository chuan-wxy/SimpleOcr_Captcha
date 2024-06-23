"""该模块为预测模块，其包含test_pred()--批量预测模块和pic_pred()--单次预测模块"""

import torch
import common
import one_hot
import my_datasets
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from colorama import Fore, Style


# 用于批量预测 ./datasets/test文件夹内的数据
def test_pred():
    test_data = my_datasets.mydatasets("./datasets/test/")  # 加载测试数据集
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)  # 创建一个数据加载器，批次大小为 1，并启用混洗
    m = torch.load("model.pth")  # 加载预训练的模型
    m.eval()  # 模型设置为评估模式

    correct = 0  # 预测正确的个数
    test_length = test_data.__len__()  # 总数居个数

    for i, (images, labels) in enumerate(test_dataloader):
        labels = labels.view(-1, common.captcha_array.__len__())
        labels_text = one_hot.vec2text(labels)  # 将 one-hot 编码的标签转换为文本形式
        output = m(images)
        output = output.view(-1, common.captcha_array.__len__())
        output_test = one_hot.vec2text(output)

        if output_test == labels_text:
            correct += 1
            print("预测正确：正确值:{},预测值:{}".format(labels_text, output_test))
        else:
            print(Fore.RED + "预测失败:正确值:{},预测值:{}".format(labels_text, output_test) + Style.RESET_ALL)

        # m(imgs)
    print("一共预测{}组数据，正确率{}".format(test_length, correct / test_length * 100))


def pic_pred(pic_path):
    img = Image.open(pic_path)
    trans = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((60, 160)),
        transforms.ToTensor()
    ])
    img_tensor = trans(img)
    img_tensor = img_tensor.reshape((1, 1, 60, 160))
    m = torch.load("model.pth")
    output = m(img_tensor)
    output = output.view(-1, common.captcha_array.__len__())

    outputs_lable = one_hot.vec2text(output)
    print(outputs_lable)


if __name__ == '__main__':
    test_pred()
    pic_pred("./datasets/test/1vgt_1718887641.png")
