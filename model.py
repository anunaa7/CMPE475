import torch
import torchvision

# The model has an input layer of 784 neurons, then another layer of 392 neuron,
# The layer after bottleneck is of 8 neurons, then the next layer is again, 392
# The output layer is 784, the same as the input layer
# The model uses relu as the activation function
# The forward function is de-structured into encode and decode functions,
# Where the encode represents the layers until the bottleneck
# and the decode represents the layers from bottleneck to output


class autoencoderMLP4Layer(torch.nn.Module):
    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
        super(autoencoderMLP4Layer, self).__init__()
        N2 = 392
        self.fc1 = torch.nn.Linear(N_input, N2)
        self.fc2 = torch.nn.Linear(N2, N_bottleneck)
        self.fc3 = torch.nn.Linear(N_bottleneck, N2)
        self.fc4 = torch.nn.Linear(N2, N_output)
        self.type = 'MLP4'
        self.input_shape = (1, 28 * 28)

    def forward(self, X):
        return self.decode(self.encode(X))

    def encode(self, X):
        X = self.fc1(X)
        X = torch.nn.functional.relu(X)
        X = self.fc2(X)
        X = torch.nn.functional.relu(X)
        return X

    def decode(self, X):
        X = self.fc3(X)
        X = torch.nn.functional.relu(X)
        X = self.fc4(X)
        X = torch.nn.functional.sigmoid(X)
        return X
