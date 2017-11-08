import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.autograd import Variable
from torch import optim
import torch.nn as nn

lr = 1e-3
batch_size = 100
epochs = 50

# 张量转换函数
trans_img = transforms.Compose([
    transforms.ToTensor()
])

trainset = MNIST('./data', train=True, transform=trans_img)
testset = MNIST('./data', train=False, transform=trans_img)

# 将输入数据集转换成张量
# trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=1)

# 构建模型
class Lenet(nn.Module):
    conv = None
    fc = None
    # 定义损失函数
    criterion = None
    # 定义优化器（梯度下降）
    optimizer = None
    cpu_mode = True

    def __init__(self, cpu_mode=True):
        super(Lenet, self).__init__()
        self.conv = nn.Sequential()
        self.conv.add_module(nn.Conv2d(1, 6, 3, stride=1, padding=1))
        self.conv.add_module(nn.MaxPool2d(2, 2))
        self.conv.add_module(nn.Conv2d(1, 6, 5, stride=1, padding=0))
        self.conv.add_module(nn.MaxPool2d(2, 2))

        self.fc = nn.Sequential()
        self.fc.add_module(nn.Linear(400, 120))
        self.fc.add_module(nn.Linear(120, 84))
        self.fc.add_module(nn.Linear(84, 10))

        self.cpu_mode = cpu_mode
        if cpu_mode:
            self.cpu()
        else:
            self.cuda()

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def compile(self, lr):
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss(size_average=False)
        # 定义优化器（梯度下降）
        self.optimizer = optim.SGD(lenet.parameters(), lr=lr)

    def fit(self, x, y, epochs, batch_size, num_workers=1, shuffle=True):
        # 训练模型
        self.train()
        for i in range(epochs):
            running_loss = 0.0
            running_acc = 0.0
            trainloader = DataLoader(x, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            for (img, label) in trainloader:
                # 如果使用CPU
                if self.cpu_mode:
                    img = Variable(img).cpu()
                    label = Variable(label).cpu()
                # 如果使用GPU
                else:
                    img = Variable(img).cuda()
                    label = Variable(label).cuda()

                # 归零操作
                self.optimizer.zero_grad()

                output = lenet(img)
                loss = self.criterion(output, label)
                # 反向传播
                loss.backward()
                self.optimizer.step()

                running_loss += loss.data[0]
                _, predict = torch.max(output, 1)
                correct_num = (predict == label).sum()
                running_acc += correct_num.data[0]

            running_loss /= len(x)
            running_acc /= len(x)
            print('[%d/%d] Loss: %.5f, Acc: %.2f' % (i + 1, epochs, running_loss, running_acc * 100))

    def evaluate(self, x):
        self.eval()
        testloss = 0.0
        testacc = 0.1
        for (img, label) in testloader:
            # 如果使用CPU
            if self.cpu_mode:
                img = Variable(img).cpu()
                label = Variable(label).cpu()
            # 如果使用GPU
            else:
                img = Variable(img).cuda()
                label = Variable(label).cuda()

            output = lenet(img)
            loss = self.criterion(output, label)
            _, predict = torch.max(output, 1)
            correct_num = (predict == label).sum()
            testacc += correct_num.data[0]

        testloss /= len(testset)
        testacc /= len(testset)
        print('Loss: %.5f, Acc: %.2f' % (testloss, testacc * 100))

    def predict(self, x):
        print('ToBe implement!')

lenet = Lenet()
lenet.compile(lr=lr)

# 训练模型
lenet.fit(x=trainset, epochs=epochs, batch_size=batch_size)


# 评估模型
lenet.evaluate(testset)