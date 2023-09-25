import datetime
from model import autoencoderMLP4Layer
import torch
import torchvision
import torchvision.datasets as data
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.data import DataLoader

# train_transform = torchvision.transforms.Compose(
#     [torchvision.transforms.ToTensor()])

# train_set = data.MNIST('./data/mnist', train=True,
#                        download=True, transform=train_transform)

# idx = int(input("enter a value between 0 and 59999 "))
# plt.imshow(train_set.data[idx], cmap='gray')
# plt.show()

# model = autoencoderMLP4Layer()
# summary(model, (1, 784))


def train(train_loader, scheduler, optimizer, model, loss_fn, n_epochs=50, device='cpu'):
    print("Training ...")
    model.train()
    losses_train = []
    loss_list = []
    epoch_numbers_x = []
    input_output_list = []
    for epoch in range(1, n_epochs + 1):
        print("epoch ", epoch)
        loss_train = 0.0
        # print(next(iter(train_loader)))
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            # for index in range(len(imgs)):
            imgs = torch.reshape(imgs, (imgs.shape[0], 784))
            outputs = model(imgs)
            input_output_list.append(
                [torch.reshape(imgs, (imgs.shape[0], 28, 28)), torch.reshape(outputs, (outputs.shape[0], 28, 28))])
            loss = loss_fn(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        # print("loss is: ", loss_train/len(train_loader))
        scheduler.step()

        losses_train += [loss_train / len(train_loader)]
        loss_list.append(loss_train / len(train_loader))
        epoch_numbers_x.append(epoch)
        print('{} Epoch {}, Training loss {}'. format(
            datetime.datetime.now(), epoch, loss_train/len(train_loader)))

    # plt.plot(epoch_numbers_x, loss_list)
    # plt.show()

    # f = plt.figure()
    # for input_output_batch in input_output_list:
    #     for input_output in input_output_batch:
    #         f.add_subplot(1, 2, 1)

    #         plt.imshow(input_output[0], cmap='gray')
    #         f.add_subplot(1, 2, 2)
    #         plt.imshow(input_output[1], cmap='gray')

    #         plt.show()
    #         # plt.clf()

    f = plt.figure()
    f.add_subplot(1, 2, 1)

    plt.imshow(input_output_list[-1][0][0], cmap='gray')
    f.add_subplot(1, 2, 2)
    plt.imshow(input_output_list[-1][0][1], cmap='gray')
    plt.show()


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

# python3 train.py -z 8 -e 50 -b 2048 -s MLP.8.pth -p loss.MLP.8.png
