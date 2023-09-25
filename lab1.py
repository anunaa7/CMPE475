import datetime
from model import autoencoderMLP4Layer
import torch
import torchvision
import torchvision.datasets as data
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.data import DataLoader

# The following function utilizes the torchsummary module
# to pretty print the information about the neural network model "autoencoderMLP4Layer"
# The function also asks the user to input a number to choose the image from the dataset and display it


def showModelSummary():
    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])
    train_set = data.MNIST('./data/mnist', train=True,
                           download=True, transform=train_transform)
    idx = int(input("Enter a value between 0 and 59999 "))
    plt.imshow(train_set.data[idx], cmap='gray')
    plt.show()
    model = autoencoderMLP4Layer()
    summary(model, (1, 784))

# This function uses the pre trained model to get the output of the model for each input image
# The model is first set to evaluation model or train(false) equivalent
# This is done so that the gradient is not calculated and back propagation does not happen in the model
# The function finally shows a sample of 5 results returned from the model in a plot


def getResults(train_loader, model, device='cpu'):
    print("Evaluating ...")
    model.eval()
    input_output_list = []
    for imgs, labels in train_loader:
        imgs = imgs.to(device=device)
        imgs = torch.reshape(imgs, (imgs.shape[0], 784))
        with torch.no_grad():
            outputs = model(imgs)
        input_output_list.append(
            [torch.reshape(imgs, (imgs.shape[0], 28, 28)), torch.reshape(outputs, (outputs.shape[0], 28, 28))])

    # Showing 5 pairs of inputs and outputs as example from the first batch in a graph
    f = plt.figure()
    for index in range(1, 10, 2):
        f.add_subplot(5, 2, index)
        plt.imshow(input_output_list[0][0][index], cmap='gray')
        f.add_subplot(5, 2, index + 1)
        plt.imshow(input_output_list[0][1][index], cmap='gray')
    plt.show()

# This function uses the pre trained model to get the output of the model for each input image
# The model is first set to evaluation model or train(false) equivalent
# This is done so that the gradient is not calculated and back propagation does not happen in the model
# Noise value is randomly selected between 0 and 1 and is attached to the image pixels
# The function finally shows a sample of 5 results returned from the model in a plot
# The plot shows 5 rows and in each row, it shows 3 images, input, input with noise, output respectively


def getResultsWithNoise(train_loader, model, device='cpu'):
    print("Evaluating with noise ...")
    model.eval()
    input_output_with_noise_list = []
    random_noise = torch.rand(1, 784)

    for imgs, labels in train_loader:
        imgs = imgs.to(device=device)
        imgs = torch.reshape(imgs, (imgs.shape[0], 784))
        image_with_noise = imgs * random_noise
        with torch.no_grad():
            output_image_with_noise = model(image_with_noise)
        input_output_with_noise_list.append(
            [torch.reshape(imgs, (imgs.shape[0], 28, 28)), torch.reshape(image_with_noise, (image_with_noise.shape[0], 28, 28)), torch.reshape(output_image_with_noise, (output_image_with_noise.shape[0], 28, 28))])

    f = plt.figure()
    for index in range(1, 12, 3):
        f.add_subplot(5, 3, index)
        plt.imshow(input_output_with_noise_list[0][0][index], cmap='gray')
        f.add_subplot(5, 3, index + 1)
        plt.imshow(input_output_with_noise_list[0][1][index], cmap='gray')
        f.add_subplot(5, 3, index + 2)
        plt.imshow(input_output_with_noise_list[0][2][index], cmap='gray')
    plt.show()

# This function is used to train the autoencoderMLP4Layer model with 50 epochs by default
# The model is trained on MNIST dataset with a learning rate of 0.001 with Adam optimizer and StepLR scheduler and MSELoss loss function
# As the epochs increase, we see that the losses_train value decreases, this shows that the modle is becoming better
# and is able to classify the images with higher accuracy. This is expected since as the epochs increase, the model is being trained more number of times
# and hence, it is able to classify the images with a higher accuracy
# The loss curve is showen in loss.MLP.8.png file
# The images are required to flattened from 28 * 28 to 1 * 784 before sending tha images as inputs to the model
# Finally, the model is saved to 'MLP.8.pth' file


def main():
    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])

    train_set = data.MNIST('./data/mnist', train=True,
                           download=True, transform=train_transform)
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
    autoEncoder = autoencoderMLP4Layer()
    autoEncoder.load_state_dict(torch.load('MLP.8.pth'))
    getResults(train_loader=train_dataloader, model=autoEncoder)
    getResultsWithNoise(train_loader=train_dataloader, model=autoEncoder)


main()

# TO RUN THIS FILE, USE THE FOLLOWING COMMAND
# python3 train.py -z 8 -e 50 -b 2048 -s MLP.8.pth -p loss.MLP.8.png
