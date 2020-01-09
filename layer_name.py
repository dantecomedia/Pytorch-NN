from collections import OrderedDict
import torch.nn as nn

input_size=784
hidden_sizes=[128,64]
output_size=10

model=nn.Sequential(OrderedDict([
	('fc1', nn.Linear(input_size, hidden_sizes[0])),
	('relu1', nn.ReLU()),
	('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
	('relu2', nn.ReLU()),
	('output', nn.Linear(hidden_sizes[1],output_size)),
	('softmax', nn.Softmax(dim=1))
	]))


print(model)
