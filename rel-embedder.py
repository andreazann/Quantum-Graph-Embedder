import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt
from networkx.algorithms import isomorphism
from minorminer import find_embedding
from datetime import datetime
import argparse
import time
import math
import numpy as np
import decimal
import os
import sys
import random
import ast

sys.path.append(os.path.abspath('/Env'))

import graphEmbEnv as gee
from updateEnvCallback import UpdateEnvCallback


from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold, EveryNTimesteps, StopTrainingOnNoModelImprovement

def argument_parser():
    """
    Get run parameters from command line
    #
    Returns: args

    # """
    CLI = argparse.ArgumentParser()

    CLI.add_argument(
        "--train1",
        type=str,
        default=None,
        help="Name of model to train for the first time")

    CLI.add_argument(
        "--train",
        type=str,
        default=None,
        help="Name of model to train again")
    
    CLI.add_argument(
        "--igraph",
        type=str,
        default=None,
        help="Input graph")
    
    CLI.add_argument(
        "--graph_set",
        type=str,
        default=None,
        help="Input graph set")

    CLI.add_argument(
        "--ts",
        type=int,
        default=100000,
        help="Timesteps")
    
    CLI.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        help="Learning rate")
        
    CLI.add_argument(
        "--ent_coef",
        type=float,
        default=0.0,
        help="Entropy coefficient")
    
    CLI.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Gamma")
    
    CLI.add_argument(
        "--norm",
        type=bool,
        default=False,
        help="Normalize heat")
    
    CLI.add_argument(
        "--avg_heat",
        type=str,
        default="aux",
        help="Include aux nodes in avg_heat")
    
    CLI.add_argument(
        "--ep_perc",
        type=float,
        default=0.1,
        help="Percentage to decrease avg_heat to end episode")
    
    CLI.add_argument(
        "--subrun",
        type=int,
        default=1,
        help="Subrun number")
    
    CLI.add_argument(
        "--algo",
        type=str,
        default="PPO",
        help="Algo either PPO or DQN")
    
    CLI.add_argument(
        "--test",
        type=str,
        default=None,
        help="Name of model to test")

    CLI.add_argument(
        "--ep",
        type=int,
        default=None,
        help="Number of episodes to test")

    args = CLI.parse_args()
    args = vars(args)

    return args


def main(name):

    launch_params = argument_parser()
    log_path = os.path.join('Training', 'Logs')

    two_node_chain_graph = nx.Graph()
    two_node_chain_graph.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7])
    two_node_chain_graph.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    three_node_chain_graph = nx.Graph()
    three_node_chain_graph.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8])
    three_node_chain_graph.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)])

    chain2x5_graph1 = nx.Graph([(0, 6), (0, 13), (1, 11), (2, 4), (2, 8), (2, 14), (3, 8), (4, 5), (4, 11), (5, 8), (5, 12), (6, 7), (6, 10), (6, 13), (7, 10), (8, 11), (8, 13), (9, 14), (11, 13), (13, 14)])
    chain2x5_graph2 = nx.Graph([(0, 4), (0, 10), (0, 13), (1, 4), (2, 8), (2, 10), (2, 11), (2, 13), (3, 4), (3, 9), (4, 7), (4, 12), (4, 13), (5, 14), (6, 13), (7, 13), (7, 14), (8, 14), (9, 13)])
    chain2x5_graph3 = nx.Graph([(0, 3), (0, 7), (0, 8), (0, 13), (1, 5), (1, 8), (2, 3), (2, 14), (3, 10), (4, 6), (4, 7), (4, 8), (4, 9), (4, 13), (5, 8), (5, 11), (5, 14), (7, 14), (8, 12), (9, 10), (12, 14)])

    n30c20_graph = nx.Graph([(0, 14), (0, 17), (0, 21), (1, 5), (1, 13), (1, 14), (1, 15), (1, 19), (1, 21), (1, 22), (1, 26), (1, 27), (1, 28), (2, 8), (2, 11), (2, 16), (2, 17), (2, 21), (2, 23), (2, 25), (3, 4), (3, 20), (3, 22), (3, 24), (3, 28), (4, 5), (4, 7), (4, 14), (4, 17), (4, 18), (4, 28), (4, 29), (5, 20), (5, 26), (6, 8), (6, 25), (6, 26), (7, 12), (7, 21), (8, 10), (8, 20), (8, 22), (8, 24), (8, 28), (8, 29), (9, 14), (9, 25), (9, 27), (10, 18), (10, 20), (10, 22), (10, 28), (11, 12), (11, 14), (11, 17), (11, 19), (11, 23), (11, 26), (11, 29), (12, 19), (12, 20), (12, 21), (12, 24), (12, 26), (13, 18), (13, 20), (13, 21), (13, 22), (13, 27), (13, 28), (14, 15), (14, 22), (14, 27), (16, 19), (16, 21), (16, 22), (16, 23), (17, 18), (17, 19), (17, 26), (18, 25), (19, 20), (19, 23), (19, 24), (20, 23), (21, 29), (22, 24), (23, 25), (23, 29), (24, 25), (24, 28), (25, 26), (25, 27)])

    chain2x5_training_set = get_graph_dataset("2node_chain", "training_set_2nodes_chain.txt")
    chain2x5_validation_set = get_graph_dataset("2node_chain", "validation_set_2nodes_chain.txt")
    chain2x5_test_set = get_graph_dataset("2node_chain", "test_set_2nodes_chain.txt")

    n30c20_training_set = get_graph_dataset("n30c20", "training_set_n30c20x5.txt")
    n30c20x10_training_set = get_graph_dataset("n30c20", "training_set_n30c20x10.txt")
    n30c20x20_training_set = get_graph_dataset("n30c20", "training_set_n30c20x20.txt")
    n30c20_validation_set = get_graph_dataset("n30c20", "validation_set_n30c20.txt")
    n30c20_test_set = get_graph_dataset("n30c20", "test_set_n30c20.txt")

    n50c20x10_training_set = get_graph_dataset("n50c20", "training_set_n50c20x10.txt")
    n50c20x20_training_set = get_graph_dataset("n50c20", "training_set_n50c20x20.txt")
    n50c20x5_validation_set = get_graph_dataset("n50c20", "validation_set_n50c20x5.txt")
    n50c20x10_test_set = get_graph_dataset("n50c20", "test_set_n50c20x10.txt")

    target_graph=dnx.chimera_graph(15, 15, 4)
    training_set = None
    validation_set = None
    test_set = None
    start_dim = 3

    print("norm ", launch_params['norm'])
    print("avg ", launch_params['avg_heat'])

    if(launch_params['igraph']=="2node"):
        print("2node")
        training_set = [two_node_chain_graph.copy()]
    elif(launch_params['igraph']=="3node"):
        print("3node")
        training_set = [three_node_chain_graph.copy()]
    elif(launch_params['igraph']=="2x5"):
        print("2x5")
        training_set = [chain2x5_graph1.copy()]
    elif(launch_params['igraph']=="n30c20"):
        print("n30c20")
        training_set = [n30c20_graph.copy()]

    if(launch_params['graph_set']=="n30c20"):
        print("n30c20 set")
        start_dim = 5
        training_set = n30c20_training_set
        validation_set = n30c20_validation_set
        test_set = n30c20_test_set
    elif(launch_params['graph_set']=="n30c20x20"):
        print("n30c20x20 set")
        start_dim = 5
        training_set = n30c20x20_training_set
        validation_set = n30c20_validation_set
        test_set = n30c20_test_set
    elif(launch_params['graph_set']=="n30c20x10"):
        print("n30c20x10 set")
        start_dim = 5
        training_set = n30c20x10_training_set
        validation_set = n30c20_validation_set
        test_set = n30c20_test_set
    elif(launch_params['graph_set']=="n50c20x20"):
        print("n50c20x20 set")
        start_dim = 10
        training_set = n50c20x20_training_set
        validation_set = n50c20x5_validation_set
        test_set = n50c20x10_test_set
    elif(launch_params['graph_set']=="n50c20x10"):
        print("n50c20x10 set")
        start_dim = 10
        training_set = n50c20x10_training_set
        validation_set = n50c20x5_validation_set
        test_set = n50c20x10_test_set

    # Wrap env in un VecEnv per parall
    #env = make_vec_env(lambda: env, n_envs=1)
        
    n_subrun = launch_params['subrun']
    ep_perc = launch_params['ep_perc']
        
    TIMESTEPS = 10
    EVAL_FREQ = 5000
    SAVE_FREQ = 10000

    if launch_params['train1']:

        # Creazione degli ambienti
        env = gee.GraphEmbEnv(training_set, target_graph, ep_perc, launch_params['norm'], launch_params['avg_heat'])
        eval_env = gee.GraphEmbEnv(validation_set, target_graph, ep_perc, launch_params['norm'], launch_params['avg_heat'])

        model = None
        model_name_clean = launch_params['train1']


        for i in range(n_subrun):

            model_name = model_name_clean+"_subrun"+str(i+1)
            model_path = os.path.join('Training', 'Saved Models', model_name)

            #checkpoint_path = os.path.join(model_path, 
            checkpoint_on_steps = CheckpointCallback(save_freq=SAVE_FREQ, save_path=model_path, verbose=2)
            #callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.1, verbose=1)
            stop_no_improv_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=20, verbose=1)
            eval_callback = EvalCallback(eval_env, eval_freq=EVAL_FREQ, callback_on_new_best=stop_no_improv_callback, verbose=1)
            global_callback_list = CallbackList([checkpoint_on_steps, eval_callback])

            if launch_params['algo'] == "PPO":
                model = PPO("MlpPolicy", env, verbose=1, learning_rate=launch_params['lr'], ent_coef=launch_params['ent_coef'], tensorboard_log=log_path)
                model.learn(total_timesteps=launch_params['ts'], progress_bar=True, callback=global_callback_list, tb_log_name=model_name)
            elif launch_params['algo'] == "DQN": 
                model = DQN("MlpPolicy", env, verbose=1, learning_rate=launch_params['lr'], gamma=launch_params['gamma'], tensorboard_log=log_path)
                model.learn(total_timesteps=launch_params['ts'], progress_bar=True, callback=global_callback_list, log_interval=512, tb_log_name=model_name)
            

            """model = None
            model_path = os.path.join('Training', 'Saved Models', launch_params['train1'])
            if launch_params['algo'] == "PPO":
                model = PPO("MlpPolicy", env, verbose=1, learning_rate=launch_params['lr'], tensorboard_log=log_path)
            elif launch_params['algo'] == "DQN": 
                model = DQN("MlpPolicy", env, verbose=1, learning_rate=launch_params['lr'], gamma=launch_params['gamma'], tensorboard_log=log_path)
            
            #checkpoint_path = os.path.join(model_path, 
            checkpoint_on_steps = CheckpointCallback(save_freq=SAVE_FREQ//TIMESTEPS, save_path=model_path, verbose=2)
            update_env_callback = UpdateEnvCallback(env, eval_env, EVAL_FREQ//TIMESTEPS, chain2x5_training_set, chain2x5_validation_set)
            n_timesteps_callback_list = CallbackList([checkpoint_on_steps, update_env_callback])
            n_timesteps_callback = EveryNTimesteps(n_steps=TIMESTEPS, callback=n_timesteps_callback_list)
            #callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.1, verbose=1)
            stop_no_improv_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=20, verbose=1)
            eval_callback = EvalCallback(eval_env, eval_freq=EVAL_FREQ, callback_on_new_best=stop_no_improv_callback, verbose=1)
            global_callback_list = CallbackList([n_timesteps_callback, eval_callback])
            if launch_params['algo'] == "PPO":
                model.learn(total_timesteps=int(1e10), progress_bar=False, callback=global_callback_list, tb_log_name=launch_params['train1'])
            elif launch_params['algo'] == "DQN": 
                model.learn(total_timesteps=int(1e10), progress_bar=False, callback=global_callback_list, log_interval=512, tb_log_name=launch_params['train1'])
            """

            """model = None
            model_path = os.path.join('Training', 'Saved Models', launch_params['train1'])
            if launch_params['algo'] == "PPO":
                model = PPO("MlpPolicy", env, verbose=1, learning_rate=launch_params['lr'], tensorboard_log=log_path)
            elif launch_params['algo'] == "DQN": 
                model = DQN("MlpPolicy", env, verbose=1, learning_rate=launch_params['lr'], gamma=launch_params['gamma'], tensorboard_log=log_path)
            graph_i = 0
            for source_graph in chain2x5_training_set:
                print(f"### TRAINING ON GRAPH {graph_i+1} ###")
                env.update_source_graph(source_graph)  # Aggiorna l'ambiente con il nuovo grafo
                if launch_params['algo'] == "PPO":
                    for i in range(1, round(launch_params['ts']/TIMESTEPS)+1):
                        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=launch_params['train1'])
                        step_path = os.path.join(model_path, str(graph_i*launch_params['ts'] + TIMESTEPS*i))
                        model.save(step_path)
                elif launch_params['algo'] == "DQN": 
                    for i in range(1, round(launch_params['ts']/TIMESTEPS)+1):
                        model.learn(total_timesteps=TIMESTEPS,  log_interval=512, reset_num_timesteps=False, tb_log_name=launch_params['train1'])
                        step_path = os.path.join(model_path, str(graph_i*launch_params['ts'] + TIMESTEPS*i))
                        model.save(step_path)
                graph_i = graph_i + 1"""
            
            """model = None
            if launch_params['algo'] == "PPO":
                model = PPO("MlpPolicy", env, verbose=1, learning_rate=launch_params['lr'], tensorboard_log=log_path)
                model.learn(total_timesteps=launch_params['ts'], progress_bar=True, tb_log_name=launch_params['train1'])
                    
            elif launch_params['algo'] == "DQN": 
                model = DQN("MlpPolicy", env, verbose=1, learning_rate=launch_params['lr'], gamma=launch_params['gamma'], tensorboard_log=log_path)
                model.learn(total_timesteps=launch_params['ts'], progress_bar=True, log_interval=512, tb_log_name=launch_params['train1'])
                    

            model_path = os.path.join('Training', 'Saved Models', launch_params['train1'])

            # Salvataggio del modello
            model.save(model_path)"""

    elif launch_params['train']:

        env = gee.GraphEmbEnv(training_set, target_graph, ep_perc, launch_params['norm'], launch_params['avg_heat'])
        eval_env = gee.GraphEmbEnv(validation_set, target_graph, ep_perc, launch_params['norm'], launch_params['avg_heat'])

        model_path = os.path.join('Training', 'Saved Models', launch_params['train'])

        start = time.time()

        model = None

        if launch_params['algo'] == "PPO":
            model = PPO.load(model_path, env=env)
            model.learn(total_timesteps=launch_params['ts'])
        elif launch_params['algo'] == "DQN":
            model = DQN.load(model_path, env=env)
            model.learn(total_timesteps=launch_params['ts'], log_interval=512)

        model.save(model_path)

        end = time.time()-start

        print("%.2f secs" % end)

    elif launch_params['test']:

        env = gee.GraphEmbEnv([test_set[0]], target_graph, ep_perc, launch_params['norm'], launch_params['avg_heat'])
        env_rnd = gee.GraphEmbEnv([test_set[0]], target_graph, ep_perc, launch_params['norm'], launch_params['avg_heat'])

        model_path = None

        if(launch_params['igraph']):
            print("igraph testing")
        elif(launch_params['graph_set']):
            model_set_folder = "set_"+launch_params['graph_set']
            model_name = launch_params['test']
            model_zip = "rl_model_100000_steps"

        model_path = os.path.join('Training', 'Saved Models', model_set_folder, model_name, model_zip)
        model = None

        if launch_params['algo'] == "PPO":
            model = PPO.load(model_path, env=env)
        elif launch_params['algo'] == "DQN":
            model = DQN.load(model_path, env=env)

        episodes = launch_params['ep']

        for subrun in range(launch_params['subrun']):

            date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-")
            date_time_fld_path = os.path.join("Testing", date_time+launch_params['test'])
            os.makedirs(date_time_fld_path, exist_ok=True)

            tot_rel_i = 0
            tot_rel_ep_len_m = 0
            tot_rel_rew_m = 0
            tot_rel_total_qubits_m = 0
        
            tot_rnd_i = 0
            tot_rnd_ep_len_m = 0
            tot_rnd_rew_m = 0
            tot_rnd_total_qubits_m = 0

            tot_true_graph_qubits_m = 0

            curr_graph = 0
            total_action_freq = {}
            init_action_freq(total_action_freq)

            for graph in test_set:

                graph_fld = os.path.join(date_time_fld_path, f"Graph {curr_graph+1}")
                env.update_source_graph_set([graph])

                rel_i = 0
                rel_ep_len_m = 0
                rel_rew_m = 0
                rel_total_qubits_m = 0
            
                rnd_i = 0
                rnd_ep_len_m = 0
                rnd_rew_m = 0
                rnd_total_qubits_m = 0

                true_graph_qubits_m = 0

                graph_action_freq = {}
                init_action_freq(graph_action_freq)

                for episode in range(1, episodes+1):
                    obs, _info = env.reset()
                    n_state, _info_rnd = env_rnd.reset()
                    terminated = False
                    terminated_rnd = False
                    score = 0
                    score_rnd = 0
                    ts = 0
                    ts_rnd = 0
                    info = {}
                    total_info = {}
                    info_rnd = {}
                    total_info_rnd = {}
                    action_freq = {}
                    
                    episode_fld = os.path.join(graph_fld, f"Episode {episode}")
                    os.makedirs(episode_fld, exist_ok=True)
                    total_info[0] = _info.copy()
                    #print("START NODES HEAT ", total_info[0]['nodes_heat'])
                    total_info_rnd[0] = _info_rnd.copy()
                    init_action_freq(action_freq)

                    
                    while not terminated:
                        action, _state = model.predict(obs)
                        obs, reward, terminated, truncated, info = env.step(action)
                        score+=reward
                        ts = ts+1
                        total_info[ts] = info.copy()
                        print("TS ", ts)
                        #print_selected(info)
                        register_action_freq(graph_action_freq, action_freq, obs, action)
                        if(terminated and (not info['invalid_ep'])):
                            rel_ep_len_m = rel_ep_len_m + ts
                            rel_rew_m = rel_rew_m + score
                            rel_i = rel_i + 1
                        #print("reward rel ", reward)
                    print('ReL\n\nEpisode:{} Score:{} Total timesteps:{}'.format(episode, score, ts))

                    print("REL EMBEDDING {}\n".format(info['modified_graph'].edges()))

                    while not terminated_rnd:
                        #action, _state = env_rnd.action_space.sample()
                        #print("STATE ", n_state.tolist())
                        action = np.array(random.randint(0, get_n_valid_actions(n_state.tolist().copy())))
                        n_state, reward, terminated_rnd, truncated, info_rnd = env_rnd.step(action)
                        score_rnd+=reward
                        ts_rnd=ts_rnd+1
                        total_info_rnd[ts_rnd] = info_rnd.copy()
                        if(terminated_rnd and (not info_rnd['invalid_ep'])):
                            rnd_ep_len_m = rnd_ep_len_m + ts_rnd
                            rnd_rew_m = rnd_rew_m + score_rnd
                            rnd_i = rnd_i + 1
                        #print("reward rnd", reward)
                    print('Rnd\n\nEpisode:{} Score:{} Total timesteps:{}'.format(episode, score_rnd, ts_rnd))

                    dim = start_dim
                    G = None
                    embedding_rel = None
                    
                    while not embedding_rel:
                        G=dnx.chimera_graph(dim, dim, 4) 
                        embedding_rel = find_embedding(info['modified_graph'], G)
                        dim = dim+1

                    print("EMBEDDED REL {}\n\nFINE EP\n\n".format(embedding_rel))
                    embedding_rel_comp = recompose_emb(embedding_rel, info['aux_nodes'])
                    #save_figs(H, G, embedding_rel, embedding_rel_comp)

                    embedding_rnd=find_embedding(info_rnd['modified_graph'], G)
                    while not embedding_rnd:
                        G=dnx.chimera_graph(dim, dim, 4) 
                        embedding_rnd = find_embedding(info_rnd['modified_graph'], G)
                        dim = dim+1
                    embedding_rnd_comp = recompose_emb(embedding_rnd, info_rnd['aux_nodes'])
                    #save_figs(H, G, embedding_rnd, embedding_rnd_comp)

                    if(not info['invalid_ep']):
                        ep_total_qubits = sum(len(l) for l in embedding_rel_comp.values())
                        rel_total_qubits_m = rel_total_qubits_m + ep_total_qubits

                    if(not info_rnd['invalid_ep']):
                        ep_total_qubits = sum(len(l) for l in embedding_rnd_comp.values())
                        rnd_total_qubits_m = rnd_total_qubits_m + ep_total_qubits

                    print(f"### TRUE GRAPH {curr_graph} ###")
                    
                    H = graph

                    embedding_true=find_embedding(H, G)
                    while not embedding_true:
                        G=dnx.chimera_graph(dim, dim, 4) 
                        embedding_true=find_embedding(H, G)
                        dim = dim+1

                    ep_total_qubits = sum(len(l) for l in embedding_true.values())
                    true_graph_qubits_m = true_graph_qubits_m + ep_total_qubits

                    episode_log(episode, total_info, total_info_rnd, score, score_rnd, action_freq, info['modified_graph'], embedding_rel, embedding_rel_comp, embedding_rnd, embedding_rnd_comp, embedding_true, ts, ts_rnd, episode_fld)
                    save_figs(H, G, embedding_rel, embedding_rel_comp, embedding_rnd, embedding_rnd_comp, embedding_true, episode_fld)

                total_test_log(graph_fld, rel_ep_len_m/rel_i, rel_rew_m/rel_i, rel_total_qubits_m/rel_i, rnd_ep_len_m/rnd_i, rnd_rew_m/rnd_i, rnd_total_qubits_m/rnd_i, true_graph_qubits_m/episodes, graph_action_freq)

                curr_graph = curr_graph + 1

                tot_rel_i += rel_i
                tot_rel_ep_len_m += rel_ep_len_m
                tot_rel_rew_m += rel_rew_m
                tot_rel_total_qubits_m += rel_total_qubits_m
            
                tot_rnd_i += rnd_i
                tot_rnd_ep_len_m += rnd_ep_len_m
                tot_rnd_rew_m += rnd_rew_m
                tot_rnd_total_qubits_m += rnd_total_qubits_m

                tot_true_graph_qubits_m += true_graph_qubits_m

                register_total_action_freq(total_action_freq, graph_action_freq)

            total_test_log(date_time_fld_path, tot_rel_ep_len_m/tot_rel_i, tot_rel_rew_m/tot_rel_i, tot_rel_total_qubits_m/tot_rel_i, tot_rnd_ep_len_m/tot_rnd_i, tot_rnd_rew_m/tot_rnd_i, tot_rnd_total_qubits_m/tot_rnd_i, tot_true_graph_qubits_m/(len(test_set)*episodes), total_action_freq)
        
    env.close()

def episode_log(episode, total_info, total_info_rnd, score, score_rnd, action_freq, rel_graph, embedding_rel, embedding_rel_comp, embedding_rnd, embedding_rnd_comp, embedding_true, ts, ts_rnd, episode_fld):

    ep_rel_total_qubits = sum(len(l) for l in embedding_rel_comp.values())
    ep_rnd_total_qubits = sum(len(l) for l in embedding_rnd_comp.values())
    ep_true_total_qubits = sum(len(l) for l in embedding_true.values())

    file_content = f"Episode: {episode}\n\n"
    file_content = f"{file_content}START HEAT: {total_info[0]['start_heat']}\n\nSTART NODES HEAT: {total_info[0]['nodes_heat']}\n\nSTART EMBEDDING: {total_info[0]['embeddings']}\n\nScore: {score}\nTotal timesteps: {ts}\nReL qubits used: {ep_rel_total_qubits}\nRnd qubits used: {ep_rnd_total_qubits}\nTrue emb qubits used: {ep_true_total_qubits}\n\nActions frequency: {action_freq}\n\nReL graph: {rel_graph.edges()}\n\nEmbedding ReL: {embedding_rel}\n\nEmbedding ReL Composed: {embedding_rel_comp}\n\n"

    for k in range(1, len(total_info)):
        neighbor_selected = ""
        if total_info[k]['action'] < len(total_info[k]['neighbors']):
            neighbor_selected = total_info[k]['neighbors'][total_info[k]['action']]
        else:
            neighbor_selected = -1
        file_content = f"{file_content}\t### TIMESTEP {k} ###\n\n\tNodes heat: {total_info[k]['nodes_heat']}\n\n\tAvg heat: {total_info[k]['avg_heat']}\n\n\tTarget heat: {total_info[k]['target_heat']}\n\n\tEmbeddings: {total_info[k]['embeddings']}\n\n\tPriority node: {total_info[k]['priority_node']}\n\n\tAction node selected: {neighbor_selected}\n\n\tAction selected: {total_info[k]['action']}\n\n\tState: {total_info[k]['state']}\n\n\tAvail aux node: {total_info[k]['avail_aux_node']}\n\n"

    file_content = f"{file_content}Score random: {score_rnd}\nTotal timesteps random: {ts_rnd}\n\nEmbedding ReL: {embedding_rnd}\n\nEmbedding ReL Composed: {embedding_rnd_comp}\n\n"

    file_path = os.path.join(episode_fld, "episode_info.txt")

    # Scrivi il contenuto nel file txt
    with open(file_path, 'w') as file:
        file.write(file_content)

def total_test_log(log_fld, rel_ep_len_m, rel_rew_m, rel_total_qubits_m, rnd_ep_len_m, rnd_rew_m, rnd_total_qubits_m, true_total_qubits_m, total_action_freq):

    testing_log_path = os.path.join(log_fld, "cumulative_info.txt")

    improv_perc = "+"+str(((rel_total_qubits_m-true_total_qubits_m)/true_total_qubits_m)*100) if rel_total_qubits_m > true_total_qubits_m else "-"+str(((true_total_qubits_m-rel_total_qubits_m)/rel_total_qubits_m)*100)
    improv_perc = improv_perc[0:5]+"%"

    file_content = f"RL\n\nep_len_mean: {rel_ep_len_m}\nrew_mean: {rel_rew_m}\nep_qubits_mean: {rel_total_qubits_m}\n\nRND\n\nep_len_mean: {rnd_ep_len_m}\nrew_mean: {rnd_rew_m}\nep_qubits_mean: {rnd_total_qubits_m}\n\nTRUE EMBEDDING\n\nep_qubits_mean: {true_total_qubits_m}\n\nActions frequency: {total_action_freq}\n\nRESULT:\n\n{improv_perc} qubits used"

    with open(testing_log_path, 'w') as file:
        file.write(file_content)


def get_n_valid_actions(state):
    n_valid_actions = 0
    for elem in state:
        if elem == 0:
            break
        n_valid_actions += 1
    return n_valid_actions

def recompose_emb(emb, aux_nodes):
    embedding = emb.copy()
    for node in aux_nodes:
        for aux_node in sorted(aux_nodes[node]):
            chain_phys_qubits = embedding[aux_node]
            del embedding[aux_node]
            embedding[node] = embedding[node] + chain_phys_qubits
    return embedding

def init_action_freq(action_freq):
    for i in range(-1, 10):
        action_freq[i] = 0

def register_total_action_freq(total_action_freq, graph_action_freq):
    for k, v in total_action_freq.items():
        total_action_freq[k] += graph_action_freq[k]

def register_action_freq(graph_action_freq, ep_action_freq, obs, action):
    n_valid_actions = get_n_valid_actions(obs.tolist().copy())
    if(action.item() < n_valid_actions):
        ep_action_freq[action.item()] = ep_action_freq[action.item()] + 1
        graph_action_freq[action.item()] = graph_action_freq[action.item()] + 1
    else:
        ep_action_freq[-1] = ep_action_freq[-1] + 1
        graph_action_freq[-1] = graph_action_freq[-1] + 1

def get_graph_dataset(dataset_folder, dataset_file_name):
    graphs_list = []
    dataset_path = os.path.join("GraphDatasets", dataset_folder, dataset_file_name)
    with open(dataset_path, 'r') as file:
        for line in file:
            edges = ast.literal_eval(line.strip())
            
            G = nx.Graph()
            G.add_edges_from(edges)
            
            graphs_list.append(G)

    return graphs_list

def save_figs(H, G, embedding_rel, embedding_rel_comp, embedding_rnd, embedding_rnd_comp, embedding_true, episode_fld):

    fig_path = os.path.join(episode_fld, 'source_graph.png')
    nx.draw(H, with_labels=True, font_weight='bold')
    plt.title("Source graph")
    plt.savefig(fig_path)
    plt.close()

    f, axes = plt.subplots(1, 1)
    fig_path = os.path.join(episode_fld, 'embedding_rel.png')
    dnx.draw_chimera_embedding(G, embedding_rel, show_labels=True)
    plt.title("Embedding ReL")
    plt.savefig(fig_path)
    #plt.show()
    plt.close()

    f, axes = plt.subplots(1, 1)
    fig_path = os.path.join(episode_fld, 'embedding_rel_comp.png')
    dnx.draw_chimera_embedding(G, embedding_rel_comp, show_labels=True)
    plt.title("Embedding ReL Composed")
    plt.savefig(fig_path)
    #plt.show()
    plt.close()

    f, axes = plt.subplots(1, 1)
    fig_path = os.path.join(episode_fld, 'embedding_rnd.png')
    dnx.draw_chimera_embedding(G, embedding_rnd, show_labels=True)
    plt.title("Embedding Random")
    plt.savefig(fig_path)
    plt.close()

    f, axes = plt.subplots(1, 1)
    fig_path = os.path.join(episode_fld, 'embedding_rnd_comp.png')
    dnx.draw_chimera_embedding(G, embedding_rnd_comp, show_labels=True)
    plt.title("Embedding Random Composed")
    plt.savefig(fig_path)
    plt.close()

    f, axes = plt.subplots(1, 1)
    fig_path = os.path.join(episode_fld, 'embedding_true.png')
    dnx.draw_chimera_embedding(G, embedding_true, show_labels=True)
    plt.title("Embedding True")
    plt.savefig(fig_path)
    plt.close()

def print_selected(info):
    keys_to_exclude = {'modified_graph', 'aux_nodes'}

    for k in info:
        if k not in keys_to_exclude:
            print("{}: {}".format(k, info[k]))
    print("")


if __name__ == '__main__':
    main('PyCharm')
