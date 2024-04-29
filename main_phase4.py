# Importing Libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
import os
from tqdm import tqdm


# Define the model, here we take resnet-18 as an example
# 定义了一个基本的构建块 VGGBlock，它用作 VGG 的基本构建块
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(VGGBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

# 定义 VGG 模型
class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            VGGBlock(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGGBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGGBlock(128, 256),
            VGGBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGGBlock(256, 512),
            VGGBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)

# 将 design_model 函数修改为返回 VGG 模型的实例
def design_model():
    return VGG()

# 定义损失函数
criterion = nn.CrossEntropyLoss()


# 训练代码
# 用于训练
def model_training(model, device, train_dataloader, optimizer, train_acc, train_losses):
    model.train()
    pbar = tqdm(train_dataloader)
    correct = 0
    processed = 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        # TODO,补全代码,填在下方

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播：计算预测值
        y_pred = model(data)

        # 计算损失
        loss = criterion(y_pred, target)

        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()

        # TODO,补全代码,填在上方

        train_losses.append(loss.item())
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        # print statistics
        running_loss += loss.item()
        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
        train_acc.append(100 * correct / processed)


# 验证代码

def model_testing(model, device, test_dataloader, test_acc, test_losses, misclassified=[]):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for index, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)

            #TODO,补全代码,填在下方

            #补全内容:获取模型输出，loss计算

            # 获取模型输出
            output = model(data)

            # 计算损失
            loss = F.cross_entropy(output, target)
            test_loss += loss.item()  # 累加损失值

            #TODO,补全代码,填在上方

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_dataloader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))

    test_acc.append(100. * correct / len(test_dataloader.dataset))


def main():
    # 设备选择（如果可用，则选择 cuda，否则选择 cpu）
    device = "cuda" if torch.cuda.is_available else "cpu"
    print(device)

    # prepare datasets and transforms
    # 数据准备：为训练和测试准备 CIFAR10 数据集，包括对训练数据的数据增强。
    train_transforms = transforms.Compose([
        #TODO,设计针对训练数据集的图像增强
        transforms.RandomResizedCrop(32),  # 随机裁剪并调整大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机颜色抖动
        transforms.RandomRotation(10),  # 随机旋转
        #TODO,写在上面
        transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
        transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))  #Normalize all the images
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
    ])

    data_dir = './data'
    trainset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transforms)
    testset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                              shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                             shuffle=False, num_workers=4)

    # Importing Model and printing Summary,默认是ResNet-18
    #TODO,分析讨论其他的CNN网络设计

    # 将 model = design_model().to(device) 修改为创建 VGG 模型实例
    model = design_model().to(device)
    summary(model, input_size=(3, 32, 32))

    # Training the model

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # 使用 Adagrad 优化器
    # optimizer = optim.Adagrad(model.parameters(), lr=0.001)
    # 使用 Adam 优化器
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=2, threshold=0.0001,
                                  threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

    train_acc = []
    train_losses = []
    test_acc = []
    test_losses = []
    model_path = './checkpoints'
    os.makedirs(model_path, exist_ok=True)

    EPOCHS = 40
    # 为固定的 epochs 数量执行训练循环，其中迭代地调用训练和测试函数。
    for i in range(EPOCHS):
        print(f'EPOCHS : {i}')
        # TODO,补全model_training里的代码
        model_training(model, device, trainloader, optimizer, train_acc, train_losses)
        scheduler.step(train_losses[-1])
        # TODO,补全model_testing里的代码
        model_testing(model, device, testloader, test_acc, test_losses)

        # 保存模型权重
        torch.save(model.state_dict(), os.path.join(model_path, 'model.pth'))

    fig, axs = plt.subplots(2, 2, figsize=(25, 20))

    axs[0, 0].set_title('Train Losses')
    axs[0, 1].set_title(f'Training Accuracy (Max: {max(train_acc):.2f})')
    axs[1, 0].set_title('Test Losses')
    axs[1, 1].set_title(f'Test Accuracy (Max: {max(test_acc):.2f})')

    axs[0, 0].plot(train_losses)
    axs[0, 1].plot(train_acc)
    axs[1, 0].plot(test_losses)
    axs[1, 1].plot(test_acc)

    # 保存图像
    # 绘图：最后，使用 matplotlib 绘制并保存训练和测试损失，以及准确度。
    plt.savefig('curves_phase4.png')  # 保存为名为 'plot.png' 的图片文件


if __name__ == '__main__':
    main()
