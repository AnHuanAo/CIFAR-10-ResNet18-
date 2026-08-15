import gc
import itertools
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def make_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMG_SIZE, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25),
    ])


def make_loader(num_workers, persistent):
    ds = torchvision.datasets.CIFAR10(
        root='./data/cifar-10-python', train=True, download=False, transform=make_transforms()
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=128, shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=persistent,
    )


def bench(name, num_workers, persistent, use_amp, n_batches=60):
    torch.cuda.empty_cache()
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to('cuda')
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    loader = make_loader(num_workers, persistent)
    it = iter(loader)

    for _ in range(3):  # warmup
        x, y = next(it)
        x, y = x.cuda(), y.cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            loss = crit(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad()

    start = time.time()
    for _ in range(n_batches):
        x, y = next(it)
        x, y = x.cuda(), y.cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            loss = crit(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    dt = time.time() - start
    print(f"{name}: {n_batches / dt:.1f} it/s | {n_batches * 128 / dt:.0f} img/s | {dt / n_batches * 1000:.0f} ms/batch")

    del loader, model, opt, crit
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    bench("A num_workers=0, FP32 (当前配置)", 0, False, False)
    bench("B num_workers=4, persistent, FP32", 4, True, False)
    bench("C num_workers=4, persistent, bf16", 4, True, True)
