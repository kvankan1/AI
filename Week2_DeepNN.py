import torch
import numpy as np

# Set the PyTorch and numpy random seeds for reproducibility:
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

def train_regressor_nn(n_features, n_hidden_neurons, learning_rate, n_epochs, X, Y):

    """ TODO: 
    Part 1:
        - create the model
        - define the loss function
        - define the optimizer
        - train the network
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, n_hidden_neurons),
        torch.nn.Sigmoid(),
        torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
        torch.nn.Sigmoid(),
        torch.nn.Linear(n_hidden_neurons, n_features)
    )

    # MSE loss function:
    loss_fn = torch.nn.MSELoss()

    # optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the network:
    for t in range(n_epochs):
        # Forward pass
        y_pred = model(X)
    # Compute and print loss. We pass Tensors containing the predicted and
    # true values of y, and the loss function returns a Tensor containing
    # the loss.
        loss = loss_fn(y_pred, Y)
        # if t % 100 == 0:
        # print(t, loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # return the trained model
    return model
