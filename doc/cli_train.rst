##############
Model training
##############

This section is related to the JUNE-NZ component: **cli_train**. It is used to train mutiple model parmaters to fit the provided target data.

**********
1. Background
**********
At the core of this model lies a SIR (``Susceptible`` -> ``Infected`` -> ``Recovered``) procedure. 
All individual agents within the model engage in interactions using a Graph Neural Network (GNN). 
The GNN facilitates the application of various policies within the simulated society, including actions like school closures and vaccination campaigns.

The model's behavior is governed by a range of parameters specified in the configuration file. 
These parameters are initially set using a Long Short-Term Memory Neural Network, and they are fine-tuned during the model's iterations through backpropagation. 
This process involves minimizing the error between the model's predictions and the desired outcomes.