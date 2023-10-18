##############
Introduction
##############

**********
The need for improved practice 
**********

Traditional agent-based models (ABMs) are built upon object-oriented programming, defining agents and actions individually for ease of understanding and design 
(e.g., EpiModel by Emory University, a widely used model for infectious disease research). 
However, this approach demands each agent to maintain its memory, potentially causing high memory usage and 
slow computation for large-scale simulations. 
Furthermore, calibrating these models can be challenging and time-consuming due to numerous parameters, 
including unobservable ones like interaction matrices in schools or workplaces.

To address these issues, researchers from _Oxford University_ and _MIT_ have jointly developed **GradABM** 
(see the details [here](https://arxiv.org/abs/2207.09714)),
a new approach that uses Graph Neural Networks (GNNs) and Long Short-Term Memory (LSTM) 
to represent agents and their interactions. **GradABM** is able to simulate large populations 
of millions of agents much faster than traditional ABMs, 
and with less memory overhead. Crucially, users do not need to 
manually specify the actual values of a large number of parameters in **GradABM**. 
Instead, they can simply provide a reasonable range of values for each parameter, 
based on previous studies and empirical experience. 
The deep learning neural network will then learn the actual values of the parameters, 
taking into account the temporal evolution of social dynamics.

**GradABM** was first developed to simulate how vaccines were given and their impacts during the COVID-19 pandemic in London. 
It has since been further developed in other countries such as the modelling of flu seasons in the US. 
ESR is working with international partners to incorporate the methodology of **GradABM** into the JUNE-NZ model, 
with the aim of developing a real-time risk and policy analysis tool for public health concerns. 
JUNE-NZ links to ESR's notifiable disease database, enabling seamless observation and modelling, 
and provides an external-facing dashboard that will allow end-users to run the model, using different scenarios.
