import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMG_SIZE, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data/cifar-10-python', train=True, download=False, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True,
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data/cifar-10-python', train=False, download=False, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True,
    )

    # 预训练骨干 + 随机分类头
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    # 冻结骨干：只有分类头可训练（线性探测）
    for p in model.parameters():
        p.requires_grad = False
    model.fc.requires_grad_(True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    def train_one_epoch():
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in tqdm(trainloader, desc='Training', mininterval=1.0):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return running_loss / len(trainloader), 100.0 * correct / total

    def test():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in tqdm(testloader, desc='Testing', mininterval=1.0):
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return 100.0 * correct / total

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(10):
        tr_loss, tr_acc = train_one_epoch()
        te_acc = test()
        scheduler.step()
        best_acc = max(best_acc, te_acc)
        print(f"Epoch {epoch+1:02d}: Train Acc {tr_acc:.2f}% | Val Acc {te_acc:.2f}% | Best {best_acc:.2f}%")
    print(f"线性探测完成，用时 {time.time()-t0:.0f}s，最佳验证准确率 {best_acc:.2f}%")


if __name__ == '__main__':
    main()
