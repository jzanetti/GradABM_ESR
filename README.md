This is the implementation of the differentiable ABM at ESR.

The algorithm is described [here](https://arxiv.org/abs/2207.09714). Codes in this repository are adapted from [here](https://github.com/AdityaLab/GradABM).

The codes here are only used to demostrate the usefulness of learnable parameters in an ABM via graph neural network. More applications
can be developed by extending this simple model.

This simple model contains _10_ agents, they are all living and interacting within the same household. 
Agents are defined in ``data/agents.csv`` (_Note that there are 10 age groups representing ages: 0-10, 11-20, ..., 80+_).
Their interaction links are defined in ``data/interaction_graph_cfg.csv``. At the begining, all the agents are not infected, while 


The learnable parameters include:

- r (The scale factor for the overal infection rate)
- mortality_rate
- initial_infections_percentage

Note that the data provided for the toy model is not sufficient to do any meaningful tunning.


Contact: sijin.zhang@esr.cri.nz