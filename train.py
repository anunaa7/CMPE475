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


def getResults(train_loader, scheduler, optimizer, model, loss_fn, n_epochs=10, device='cpu'):
    print("Evaluating ...")
    model.eval()
    input_output_list = []
    input_output_with_noise_list = []
    random_noise = torch.rand(1, 784)
    noise_images = []
    for epoch in range(1, n_epochs + 1):
        print("epoch ", epoch)
        input_output_list = []
        # print(next(iter(train_loader)))
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            # for index in range(len(imgs)):
            imgs = torch.reshape(imgs, (imgs.shape[0], 784))
            with torch.no_grad():
                outputs = model(imgs)
            # image_with_noise = torch.mul(imgs, random_noise)
            image_with_noise = imgs * random_noise
            noise_images.append(torch.reshape(
                image_with_noise, (image_with_noise.shape[0], 28, 28)))
            output_image_with_noise = model(image_with_noise)
            input_output_list.append(
                [torch.reshape(imgs, (imgs.shape[0], 28, 28)), torch.reshape(outputs, (outputs.shape[0], 28, 28))])
            input_output_with_noise_list.append(
                [torch.reshape(imgs, (imgs.shape[0], 28, 28)), torch.reshape(image_with_noise, (image_with_noise.shape[0], 28, 28)), torch.reshape(output_image_with_noise, (output_image_with_noise.shape[0], 28, 28))])
            loss = loss_fn(outputs, imgs)

    f = plt.figure()

    for index in range(1, 10, 2):
        # f.add_subplot(1, 2, 1)
        f.add_subplot(5, 2, index)

        plt.imshow(input_output_list[0][0][index], cmap='gray')
        # f.add_subplot(1, 2, 2)
        f.add_subplot(5, 2, index + 1)
        plt.imshow(input_output_list[0][1][index], cmap='gray')
    plt.show()


def getResultsWithNoise(train_loader, scheduler, optimizer, model, loss_fn, n_epochs=2, device='cpu'):
    print("Evaluating ...")
    model.eval()
    input_output_list = []
    input_output_with_noise_list = []
    random_noise = torch.rand(1, 784)
    noise_images = []
    for epoch in range(1, n_epochs + 1):
        print("epoch ", epoch)
        input_output_list = []
        # print(next(iter(train_loader)))
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            # for index in range(len(imgs)):
            imgs = torch.reshape(imgs, (imgs.shape[0], 784))
            with torch.no_grad():
                outputs = model(imgs)
            # image_with_noise = torch.mul(imgs, random_noise)
            image_with_noise = imgs * random_noise
            noise_images.append(torch.reshape(
                image_with_noise, (image_with_noise.shape[0], 28, 28)))
            with torch.no_grad():
                output_image_with_noise = model(image_with_noise)
            input_output_list.append(
                [torch.reshape(imgs, (imgs.shape[0], 28, 28)), torch.reshape(outputs, (outputs.shape[0], 28, 28))])
            input_output_with_noise_list.append(
                [torch.reshape(imgs, (imgs.shape[0], 28, 28)), torch.reshape(image_with_noise, (image_with_noise.shape[0], 28, 28)), torch.reshape(output_image_with_noise, (output_image_with_noise.shape[0], 28, 28))])
            loss = loss_fn(outputs, imgs)

    f = plt.figure()

    for index in range(1, 12, 3):
        # f.add_subplot(1, 2, 1)
        f.add_subplot(5, 3, index)

        plt.imshow(input_output_with_noise_list[0][0][index], cmap='gray')
        # f.add_subplot(1, 2, 2)
        f.add_subplot(5, 3, index + 1)
        plt.imshow(input_output_with_noise_list[0][1][index], cmap='gray')

        f.add_subplot(5, 3, index + 2)
        plt.imshow(input_output_with_noise_list[0][2][index], cmap='gray')
    plt.show()


def train(train_loader, scheduler, optimizer, model, loss_fn, n_epochs=2, device='cpu'):
    print("Training ...")
    model.eval()
    losses_train = []
    loss_list = []
    epoch_numbers_x = []
    input_output_list = []
    input_output_with_noise_list = []
    random_noise = torch.rand(1, 784)
    noise_images = []
    for epoch in range(1, n_epochs + 1):
        print("epoch ", epoch)
        loss_train = 0.0
        epoch_numbers_x = []
        input_output_list = []
        # print(next(iter(train_loader)))
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            # for index in range(len(imgs)):
            imgs = torch.reshape(imgs, (imgs.shape[0], 784))
            outputs = model(imgs)
            # image_with_noise = torch.mul(imgs, random_noise)
            image_with_noise = imgs * random_noise
            noise_images.append(torch.reshape(
                image_with_noise, (image_with_noise.shape[0], 28, 28)))
            output_image_with_noise = model(image_with_noise)
            input_output_list.append(
                [torch.reshape(imgs, (imgs.shape[0], 28, 28)), torch.reshape(outputs, (outputs.shape[0], 28, 28))])
            input_output_with_noise_list.append(
                [torch.reshape(imgs, (imgs.shape[0], 28, 28)), torch.reshape(image_with_noise, (image_with_noise.shape[0], 28, 28)), torch.reshape(output_image_with_noise, (output_image_with_noise.shape[0], 28, 28))])
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

    f = plt.figure()
    print("input_output_list ", len(input_output_list))
    print("input_output_list[0] ", len(input_output_list[0]))
    print("input_output_list[0][0] ", len(input_output_list[0][0]))
    print("input_output_list[0][1] ", len(input_output_list[0][1]))
    print("input_output_list[0][0][1] ", len(input_output_list[0][0][1]))
    print("input_output_list[0][1][1] ", len(input_output_list[0][1][1]))
    print("input_output_list[0][0][1] ", (input_output_list[0][0][1]))
    print("input_output_list[0][1][1] ", (input_output_list[0][1][1]))

    for index in range(1, 10, 2):
        # f.add_subplot(1, 2, 1)
        f.add_subplot(5, 2, index)

        plt.imshow(input_output_list[0][0][index], cmap='gray')
        # f.add_subplot(1, 2, 2)
        f.add_subplot(5, 2, index + 1)
        plt.imshow(input_output_list[0][1][index], cmap='gray')
    plt.show()

    # f = plt.figure()
    # for index in range(1, 10, 3):
    #     f.add_subplot(5, 3, index)
    #     plt.imshow(input_output_list[0][0][index], cmap='gray')
    #     f.add_subplot(5, 3, index + 1)
    #     plt.imshow(input_output_list[0][0][index], cmap='gray')
    # plt.show()


train_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()])

train_set = data.MNIST('./data/mnist', train=True,
                       download=True, transform=train_transform)
train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
autoEncoder = autoencoderMLP4Layer()
autoEncoder.load_state_dict(torch.load('MLP.8.pth'))
# autoEncoder = torch.load('MLP.8.pth')
optimizer = torch.optim.Adam(autoEncoder.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1)
autoEncoder.eval()
loss_fn = torch.nn.MSELoss()
# train(train_loader=train_dataloader, scheduler=scheduler,
#       optimizer=optimizer, model=autoEncoder, loss_fn=loss_fn)

# getResults(train_loader=train_dataloader, scheduler=scheduler,
#            optimizer=optimizer, model=autoEncoder, loss_fn=loss_fn)

getResultsWithNoise(train_loader=train_dataloader, scheduler=scheduler,
                    optimizer=optimizer, model=autoEncoder, loss_fn=loss_fn)

# torch.save(autoEncoder.state_dict(), 'MLP.8.pth')

# python3 train.py -z 8 -e 50 -b 2048 -s MLP.8.pth -p loss.MLP.8.png
