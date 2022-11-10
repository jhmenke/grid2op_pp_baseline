# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:16:34 2022

@author: knarasimha

"""

import copy
import grid2op
from grid2op.Backend import PandaPowerBackend
from l2rpn_baselines.PandapowerOPFAgent import PandapowerOPFAgent
import pandas as pd
import os
# import numpy as np


# Perform a grid2op step loop until JUST before thr agent performed action
def grid2op_loop(max_iter=0):
    done = False
    obs = env.current_obs
    reward = 0.0
    step = 0
    action_sum = 0
    while done is False and step < (max_iter + 1):
        agent_action = opf_agent.act(obs, reward, done)
        action_sum = sum(agent_action.set_line_status) + \
            sum(agent_action.redispatch)
        obs, reward, done, info = env.step(agent_action)
        print(step)
        step += 1
        if action_sum > 0:
            print("Agent performed at least one action")
    return obs, reward, done, info, agent_action


# Re-load the env,obs,reward,done,agent state of a specific step and
# perform the action on next step repeatedly
def dataframe_agent_action(iteration, obs_previous, reward_previous,
                           done_previous):
    gen_1_0 = []
    gen_2_1 = []
    gen_0_4 = []
    for i in range(iteration):
        opf_agent.load_state()
        env_previous = copy.deepcopy(env)

        agent_action_next = opf_agent.act(obs_previous, reward_previous,
                                          done_previous)
        obs_next, reward_next, done_next, info_next = env_previous.step(
            agent_action_next)

        # Save changed line ids during each iteration
        # adict = agent_action_next.as_dict()
        gen_1_0.append(agent_action_next.redispatch[0])
        gen_2_1.append(agent_action_next.redispatch[1])
        gen_0_4.append(agent_action_next.redispatch[4])

    # form a dataframe from the 4 lists. each list as each column data.
    df = pd.DataFrame({'gen_1_0': gen_1_0,
                       'gen_2_1': gen_2_1,
                       'gen_0_4': gen_0_4})

    return df, agent_action_next


# Compare each rows of dataframe for similarity
def df_compare():
    if ((df_result['gen_1_0'].values == df_result['gen_1_0'][0]).all()
       and (df_result['gen_2_1'].values == df_result['gen_2_1'][0]).all()
       and (df_result['gen_0_4'].values == df_result['gen_0_4'][0]).all()):
        print('All gen redispatch action actions were equal!')
    else:
        print('At least one gen redispatch action was different!')


'''
main program
'''

# Create a new seeded environment instance and set its SEED
backend = PandaPowerBackend()
env = grid2op.make(dataset="rte_case14_redisp", backend=backend)
env.seed(5)
time_step_before_action = 3

my_grid_path = os.path.join("C:", os.sep, "ProgramData", "Anaconda3", "envs", 
                            "grid2op", "Lib", "site-packages", "grid2op", 
                            "data", "rte_case14_redisp", "grid.json")
my_savestate_path = os.path.join("C:", os.sep, "Users", "kjurczyk", 
                                 "Documents", "Kristina", "ROSALIE", 
                                 "OPF_agent")

     
# Create a seeded agent instance and set its SEED
opf_agent = PandapowerOPFAgent(env.action_space,
                               grid_path=my_grid_path,
                               savestate_path=my_savestate_path)
opf_agent.seed(3)

# Run grid2op until just before the agent performs an action and save the state
obs_before, reward_before, done_before, info_before, agent_action_before = \
    grid2op_loop(max_iter=time_step_before_action)

opf_agent.save_state()

# Repeat the time step with re-loaded agent save state and compare actions
df_result, agent_action_next = dataframe_agent_action(
    iteration=2,
    obs_previous=obs_before,
    reward_previous=reward_before,
    done_previous=done_before)

# compare the actions of the repeated time steps
df_compare()
