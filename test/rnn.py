import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch import rand as torch_rand
from torch import sigmoid
from torch import sqrt as torch_sqrt
from torch import stack as torch_stack
from torch import tensor as torch_tensor
from torch import zeros as torch_zeros
from torch.distributions import Gamma as torch_gamma
from torchmetrics.regression import MeanSquaredError


class TestModel:
    def __init__(self):
        self.init_x = 1.0

    def step(self, params, t):
        a = params[0]
        b = params[1]
        c = params[2]

        gamma_dist = torch_gamma(concentration=a, rate=1 / b)

        target = gamma_dist.log_prob(torch_tensor(t)).exp()

        target = c * target

        return target


class SimpleRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=60):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # self.fc = nn.Linear(hidden_size, output_size)

        # Apply Xavier initialization to the RNN weights
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                init.xavier_uniform_(param)

        fc = [
            nn.Linear(in_features=hidden_size, out_features=int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(in_features=int(hidden_size / 2), out_features=3),
            # nn.ReLU(),
            # nn.Linear(in_features=3, out_features=1),
        ]
        self.fc = nn.Sequential(*fc)
        self.min_values = torch_tensor([0.1, 0.1, 0.1])
        self.max_values = torch_tensor([50.0, 50.0, 300.0])
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        h0 = torch_zeros(1, x.shape[0], self.hidden_size)
        out, _ = self.rnn(torch_tensor(x).float(), torch_tensor(h0))
        return self.fc(out)

    def scale_param(self, x):
        return self.min_values + (self.max_values - self.min_values) * self.sigmod(x)


class LearnableParams(nn.Module):
    """doesn't use data signals"""

    def __init__(self):
        super().__init__()
        self.learnable_params = nn.Parameter(torch_rand(3))
        self.min_values = torch_tensor([0.1, 0.1, 0.1])
        self.max_values = torch_tensor([13.0, 15.0, 50.0])
        self.sigmod = nn.Sigmoid()

    def forward(self):
        return self.min_values + (self.max_values - self.min_values) * self.sigmod(
            self.learnable_params
        )


# -------------------------------------
# Job starts
# -------------------------------------

use_rnn_as_para = True
x = pd.read_csv("data/exp1/targets_test2_with_noise.csv").values
x = np.expand_dims(x, axis=0)
y = pd.read_csv("data/exp1/targets_test2.csv").values

if use_rnn_as_para:
    param_model = SimpleRNN()
else:
    param_model = LearnableParams()

optimizer = optim.Adam(param_model.parameters(), lr=0.01)
loss_func = MeanSquaredError()

total_timesptes = 70

my_model = TestModel()
for epoch in range(1500):
    predictions = []
    prediction = None
    optimizer.zero_grad()
    if use_rnn_as_para:
        param_values_all = param_model.forward(x)
    else:
        param_values = param_model.forward()

    for time_step in range(total_timesptes):
        if use_rnn_as_para:
            param_values = param_model.scale_param(param_values_all[0, time_step, :])

        prediction = my_model.step(
            param_values,
            time_step,
        )
        predictions.append(prediction)
    predictions = torch_stack(predictions, 0).reshape(1, -1)  # num counties, seq len
    loss = loss_func(torch_tensor(np.transpose(y)), predictions)

    loss.backward()
    optimizer.step()

    # Track the running loss
    loss_value = torch_sqrt(loss.detach()).item()

    print(f"{epoch}: {loss_value}")

plt.plot(predictions[0, :].tolist(), label="pred")
plt.plot(np.transpose(y)[0, :], label="target")
plt.legend()
plt.savefig("test.png")
plt.close()

raise Exception("!23123")

for epoch in range(200):
    running_loss = 0.0
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = param_model.forward(x)

    # Compute the loss
    loss = loss_func(torch_tensor(np.transpose(y)), outputs[:, :, 0])

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Track the running loss
    loss_value = torch_sqrt(loss.detach()).item()

    if epoch % 50 == 0:
        plt.plot(outputs[0, :, 0].tolist(), label=f"{epoch}")
    print(f"{epoch}: {loss_value}")

plt.plot(outputs[0, :, 0].tolist(), label="pred", linewidth=3)
plt.plot(y[:, 0], label="target", linewidth=3)
plt.legend()
plt.savefig("test.png")
plt.close()
