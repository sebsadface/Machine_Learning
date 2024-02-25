# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.W0 = Parameter(torch.Tensor(h, d))
        self.b0 = Parameter(torch.Tensor(h))
        self.W1 = Parameter(torch.Tensor(k, h))
        self.b1 = Parameter(torch.Tensor(k))

        sqrt_k = math.sqrt(1 / self.W0.size(1))
        self.W0.data.uniform_(-sqrt_k, sqrt_k)
        self.b0.data.zero_()
        sqrt_k = math.sqrt(1 / self.W1.size(1))
        self.W1.data.uniform_(-sqrt_k, sqrt_k)
        self.b1.data.zero_()

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        return relu(x @ self.W0.T + self.b0) @ self.W1.T + self.b1


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.W0 = Parameter(torch.Tensor(h0, d))
        self.b0 = Parameter(torch.Tensor(h0))
        self.W1 = Parameter(torch.Tensor(h1, h0))
        self.b1 = Parameter(torch.Tensor(h1))
        self.W2 = Parameter(torch.Tensor(k, h1))
        self.b2 = Parameter(torch.Tensor(k))

        sqrt_h0 = math.sqrt(1 / self.W0.size(1))
        self.W0.data.uniform_(-sqrt_h0, sqrt_h0)
        self.b0.data.zero_()
        sqrt_h1 = math.sqrt(1 / self.W1.size(1))
        self.W1.data.uniform_(-sqrt_h1, sqrt_h1)
        self.b1.data.zero_()
        sqrt_k = math.sqrt(1 / self.W2.size(1))
        self.W2.data.uniform_(-sqrt_k, sqrt_k)
        self.b2.data.zero_()

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        return relu(relu(x @ self.W0.T + self.b0) @ self.W1.T + self.b1) @ self.W2.T + self.b2


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    model.train()
    losses = []
    accuracy = 0

    while accuracy < 99:
        epoch_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        accuracy = (correct / total) * 100
        epoch_loss /= total
        losses.append(epoch_loss)
        print(f'Epoch Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.2f}%')

    return losses


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    train_dataset = TensorDataset(x, y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # F1
    f1_model = F1(h=64, d=784, k=10)
    f1_optimizer = Adam(f1_model.parameters(), lr=0.001)
    f1_losses = train(f1_model, f1_optimizer, train_loader)

    f1_test_loss, f1_accuracy = evaluate(f1_model, test_loader)
    print(f'F1 Model Test Loss: {f1_test_loss:.4f} | Test Accuracy: {f1_accuracy:.2f}%')

     # F2
    f2_model = F2(h0=32, h1=32, d=784, k=10)
    f2_optimizer = Adam(f2_model.parameters(), lr=0.001)
    f2_losses = train(f2_model, f2_optimizer, train_loader)

    f2_test_loss, f2_accuracy = evaluate(f2_model, test_loader)
    print(f'F2 Model Test Loss: {f2_test_loss:.4f} | Test Accuracy: {f2_accuracy:.2f}%')
    
    f1_params = sum(p.numel() for p in f1_model.parameters())
    f2_params = sum(p.numel() for p in f2_model.parameters())
    print(f'Total number of parameters in F1: {f1_params}')
    print(f'Total number of parameters in F2: {f2_params}')

    plt.figure(figsize=(10, 5))
    plt.plot(f1_losses, label='F1 Loss')
    plt.title('F1')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(f2_losses, label='F2 Loss')
    plt.title('F2')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def evaluate(model: Module, test_loader: DataLoader):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            loss = cross_entropy(outputs, y)
            test_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == y).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

if __name__ == "__main__":
    main()
