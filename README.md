# PandapowerOPFAgent
Baseline agent for usage of pandapower in Grid2Op

# Authors
 - Jan-Hendrik Menke \<jan-hendrik.menke@uni-kassel.de\>
 - Florian Schäfer \<florian.schaefer@uni-kassel.de\>

# Description
This agent can be seen as a template to get started with using pandapower functions in Grid2Op. While the agent can and 
does run on its own, especially the pypower optimal power flow (OPF) will likely fail to run to the end. The 
PowerModels.jl OPF implementation should make it to the end at least for the IEEE14 grid.

Using the available function in pandapower, e.g., topology analysis, power flow, or optimal power flow like in this
example may help you to employ model-based reinforcement learning, which could potentially increase your agent's score. 
To make this process more convenient, this agent will automatically parse the latest data from the observation into the
internal pandapower grid model, on which all further pandapower functions can be used. In this example, the results from
the OPF are also converted to the Grid2Op action space and sent to the environment.

# Publications
L. Thurner, A. Scheidler, F. Schäfer, J.-H. Menke et. al: pandapower - an Open Source Python Tool for Convenient 
Modeling, Analysis and Optimization of Electric Power Systems, IEEE Transaction on Power Systems Vol. 33, Issue 5, 2018
