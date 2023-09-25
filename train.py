import datetime
from model import autoencoderMLP4Layer
import torch
import torchvision
import torchvision.datasets as data
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def train(train_loader, scheduler, optimizer, model, loss_fn, n_epochs=50, device='cpu'):
    print("Training ...")
    model.train()
    losses_train = []
    loss_list = []
    epoch_numbers_x = []
    for epoch in range(1, n_epochs + 1):
        print("epoch ", epoch)
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            imgs = torch.reshape(imgs, (imgs.shape[0], 784))
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step()

        losses_train += [loss_train / len(train_loader)]
        loss_list.append(loss_train / len(train_loader))
        epoch_numbers_x.append(epoch)
        print('{} Epoch {}, Training loss {}'. format(
            datetime.datetime.now(), epoch, loss_train/len(train_loader)))

    plt.plot(epoch_numbers_x, loss_list)
    plt.savefig('loss.MLP.8.png')
    torch.save(model.state_dict(), 'MLP.8.pth')


train_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()])

train_set = data.MNIST('./data/mnist', train=True,
                       download=True, transform=train_transform)
train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
autoEncoder = autoencoderMLP4Layer()
optimizer = torch.optim.Adam(autoEncoder.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1)
loss_fn = torch.nn.MSELoss()

train(train_loader=train_dataloader, scheduler=scheduler,
      optimizer=optimizer, model=autoEncoder, loss_fn=loss_fn)
