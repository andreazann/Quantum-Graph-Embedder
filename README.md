# Single-Agent Graph Embedding for Quantum Computers via Reinforcement Learning - Master Thesis @ Politecnico di Milano

pytorch > gym > stablebaselines3 

Full thesis at [Thesis_Complete.pdf](https://github.com/user-attachments/files/17742457/Thesis_Complete.pdf)

Thesis summary at [Thesis_Executive_Summary.pdf](https://github.com/user-attachments/files/17742460/Thesis_Executive_Summary.pdf)

Supercomputers alone are insufficient for tackling NP-hard problems, one of them is the Graph Minor Embedding problem, crucial for quantum computing with D-Wave systems. 
This work focuses on using Reinforcement Learning (RL) techniques to enhance heuristics like D-Wave's minorminer for embedding problems on a Quantum Processing Unit (QPU). 
The study explores PPO and DQN RL models to optimize embeddings, aiming to reduce the number of qubits needed. Performance comparisons are made against the state-of-the-art, 
with evaluations on datasets of 15, 30, and 50-node graphs.

# Instructions for use

1. Install conda packages from this requirements file > [requirements.txt](https://github.com/user-attachments/files/17758436/requirements.txt) (stable-baselines3[extra] needs to be installed. `pip install stable-baselines3[extra]`)

3. Launch with `python rel-embedder.py [args]`

## Arguments for training a model

* --train1: this parameter is used when we want train a new model, pass a string representing the name of the model that will be saved. 
* --graph_set: when using this parameter when can pass a graphSet on which to train the model. Available sets and associated strings are "n30c20x10" (set of 10 graphs with 30 nodes and 20% of connectivity), "n30c20x20", "n50c20x10", "n50c20x20" with the same logic
* --ts: number of timesteps to train the model. Default is 100000
* --lr: with this parameter we can set the learning rate. Default is 0.0003
* --ent_coef: accepts a float that represents the entropy coefficient the model. Default is 0
* --gamma: sets the gamma parameter of the model. Default is 0.99
* --norm: if set to true, the heat values will be normalized to a float between 0 and 1. Otherwise the absolute values will be kept. Default is False
* --avg_heat: when set to true, the heat value of auxiliary nodes is included into the calculation of the average heat for the graph. Default is True
* --ep_perc: with this parameter we can set the percentage of heat to reach in order to end the episode. Default is 0.1, 10%
* --subrun: number of training runs to perform with the same settings. Default is 1
* --algo: accepts either strings "PPO" or "DQN" to set the algorithm for the training. Default is "PPO"

## Arguments for testing a model

* --test: used when testing a model. The name of the model is passed to this parameter.
* --graph_set: when using this parameter when can pass a graph_set on which to test the model
* --ep: sets the total number of episodes per run
* --subrun: number of testing runs to perform on the same model

## 1.1 Graph Embedding in Quantum Computing

D-Wave’s Quantum Processing Unit (QPU) architecture is at the base of D-Wave’s Quantum Computer exploiting the principles of quantum annealing and is suited to solve optimization problems. 
The D-Wave QPU treated in this work, composed of superconducting qubits, is organized in what is known as a Chimera graph pattern, shown below.

<img width="217" alt="chimera-unit" src="https://github.com/user-attachments/assets/f654456d-bb98-4fc0-8b6a-cc2d735a90df">
      
<img width="406" alt="chimera-lattice" src="https://github.com/user-attachments/assets/4fb2612b-cd5a-451e-a1dc-721b54a74310">

## 1.2 Brief introduction to the minorminer heuristic

The minorminer heuristic is designed to address the complex challenge of identifying a smaller graph H as a minor of a larger graph G. Given the NP-hard nature of this problem, traditional 
exact solutions become computationally infeasible for graphs of substantial size. Heuristic logic explained here <https://arxiv.org/abs/1406.2741>

# 2 Designing RL Environments for Graph Embedding

The main goal of our agent is not to fully embed a graph onto the D-Wave’s Quantum Processing Unit (QPU), but rather to **ease the embedding procedure** for the minorminer heuristic.
It is accomplished by passing in input to the heuristic a modified version of the original graph such that the nodes identified that require more qubits to be embedded onto the QPU will be
preemptively expanded into structures know as **chains**.

## 2.1 Definition of a graph complexity metric based on nodes’ heat

**Nodes’ heat** gives an indicative measure of **how many qubits are necessary to embed the current graph into a QPU**. In this way we’re able to understand at each timestep and after each modification done by the agent, 
how well the graph is approaching a target average heat.

## 2.2 Definition of local embeddings

**Local embeddings** are smaller embeddings calculated on subsets of the input graph in order to provide the heat function described in 2.1 with the number of qubits needed to embed each node onto the QPU. 
A local embedding is calculated on the subset graph such that a node is part of the local embedding if it is the node in examination or a node from a **1-hop distance** from the node in examination.

<img width="334" alt="local-embeddings" src="https://github.com/user-attachments/assets/9c8dfa9d-af39-47db-93ef-363469b80754">

## 2.3 Observation space definition

What the agent will be able to see is an **array of heat values** that are related to the nodes adjacent to the node select. The idea is that the agent will learn to select adjacent nodes that
will lower the average heat of the graph. What happens when the agent selects a node will be explained in Subsection 2.4.

<img width="422" alt="observation-space" src="https://github.com/user-attachments/assets/0d4fb58e-c51b-4e2c-a8c9-5b8387e70e4d">

## 2.4 Action definition

The agent will be able to take **discrete actions**, defined as a Discrete space in the Gym Custom Environment with the same length of 10 elements as the Observation Space described in 2.3. 
When an agent selects an action corresponding to an integer value ranging from 0 to 9 it means that it is selecting the index of the i-th node listed in the Observation Space.

<img width="498" alt="action-space" src="https://github.com/user-attachments/assets/ab44845a-34db-4629-8c2b-14a46a3f9838">

## 2.5 Node expansion process

Once we have a priority node selected as the node with the highest heat and a node selected by the agent which is adjacent to the priority node, we can proceed with the **Node Expansion** process. 
The arc shared by the priority node and the adjacent node is removed and an auxiliary node is added to the priority node. In this way initial chains are formed and the overall heat of the graph is lowered step by step.

<img width="408" alt="node-expansion" src="https://github.com/user-attachments/assets/779478df-d21d-4005-9250-6554ab0d043c">

## 2.6 Reward relevant to graph embedding

The ideal **reward** would be a reward that at each timestep can show the **progress** of the process of lowering the heat of the graph and that is **invariant to the graph** that is **analyzed** by the agent. 
For this reason the reward function has been defined as: `log(prevAvgHeat/currAvgHeat)`
Where *prevAvgHeat* represents the average heat of the graph at timestep - 1, while currAvgHeat represents the average heat of the graph at the current timestep.

## 2.7 Episode termination condition

As stated in 2.6 the reward function is expressed as a progress at each timestep of the difference of heat with respect to the previous timestep, it would be convenient to base the episode termination
condition on the same concept. In fact if we set this condition as a percentage of heat to reach we will obtain a total reward for each episode that is very similar between graphs having
different complexities. Given that we have a starting average heat defined as startAvgHeat, a current average heat as currAvgHeat and a percentage of progress of i.e. 10% passed as parameter
to the agent, the episode termination condition will be defined as: `currAvgHeat ≤ startAvgHeat ∗ (1 − 0.1)`

## 2.8 Definition of Single-Traveler agent on a graph

The concept proposed consists of having a **single agent** able only to view a **neighborhood of nodes** at each timestep. Hopping between neighborhoods where the heat higher, lowering
it using the **Node Expansion** technique and processing a new neighborhood on the next timestep. In this way the agent will have a limited view of the graph, keeping the elements
of the observation space contained while updating the parameters on the same agent improving it at each timestep. The following pseudocode shows the general behavior of the RL agent,
identified in a big portion of the step function of the Custom Gym Environment:

<img width="607" alt="single-agent-algorithm" src="https://github.com/user-attachments/assets/9d484344-2200-4e7f-a871-5c516105fc77">

# Custom parameters introduced to improve performance of RL model

* **Episode percentage**: this parameter, expressed as a percentage in terms of heat to reach, sets for the agent a limit that terminates the episode.
* **Average heat**: boolean parameter that if set to False it will exclude the heat of auxiliary nodes when calculating the average heat of the graph.
* **Normalized heat**: boolean parameter if set to true all the heat values are normalized, otherwise their absolute value is kept.

## Example of embedding optimized via RL

For ease of understanding let’s first consider a graph taken from the dataset of graphs composed by 15 nodes and on average 5 chains of 2 qubits. The RL agent will take in input the source graph show below and will start forming initial chains at each timestep. 

<img width="357" alt="single-agent-algorithm" src="https://github.com/user-attachments/assets/8a69486c-65be-4c21-8967-7613534efd67">

Once the episode is concluded, the resulting graph will be passed to the minorminer heuristic. The output produced will be the following:

<img width="357" alt="single-agent-algorithm" src="https://github.com/user-attachments/assets/7031eb0d-888a-43c2-9df9-cca2a2603714">

Nodes having the same coloring belong to the same chain. Labeled nodes with indices represent the starting node of each chain, while nodes without an index represent nodes added to the
chain. In this case the RL model added to the embedding of 15 chains numered from 0 to 14, the new chain 16 composed of one node.   
Now we will proceed to assign the chain formed by the RL agent to the original nodes to which it should be assigned.

<img width="357" alt="single-agent-algorithm" src="https://github.com/user-attachments/assets/d69d1165-775a-42ab-9559-3316e2f03aaf">

We notice how the node with index 16 has been reassigned to node 0, changing color and being part of the chain of node 0 now. 

What the minorminer heuristic has produced without the help of the RL model, is the embedding below:

<img width="357" alt="single-agent-algorithm" src="https://github.com/user-attachments/assets/2d18e404-7745-4b14-9b7e-0717593aac7b">

Assessing the result for this example, we have that the RL model+heuristic performed better then the heuristic alone, needing 20 qubits compared to the 23 qubits needing by minorminer.  

Another example showing the same steps but with bigger graphs is shown below.  

Source graph passed to the RL model:

<img width="357" alt="single-agent-algorithm" src="https://github.com/user-attachments/assets/42ce61a8-65c7-49f0-bf9e-060faeae1739">

Output graph from the RL model + heuristic:

<img width="357" alt="single-agent-algorithm" src="https://github.com/user-attachments/assets/22377402-40a8-4034-93ca-3af858330cdf">

Assigning chains formed by the RL model:

<img width="357" alt="single-agent-algorithm" src="https://github.com/user-attachments/assets/8f2f6340-662a-4081-8701-980fd12dfaff">

Output graph from the minorminer heuristic alone:

<img width="357" alt="single-agent-algorithm" src="https://github.com/user-attachments/assets/57460363-040b-4cac-9c7d-f510f5b7a953">

## Model results

The cumulative reward of the the models just analyzed is shown in the figure below: PPO model trained on dataset n30c20, DQN model trained on dataset
n30c20, PPO model trained on dataset n50c20.

<img width="557" alt="single-agent-algorithm" src="https://github.com/user-attachments/assets/b0e6802c-a6eb-47f8-9d63-4a940c5dbb24">

We notice how the DQN on n30c20 model reaches a lower reward at convergence than PPO on n30c20, and how the PPO model trained on the n50c20 dataset apparently reaches a good reward in spite of the drop in performance. This would support the idea that the algorithm managing the generation of auxiliary nodes doesn’t scale well on large graphs by creating nodes that make the minorminor heuristic underperform and use more qubits
than necessary even tough the agent performed well by obtaining a good reward and thus lowering the heat of the graph efficiently.



For an Experimental setup overview, more in depth quantitative and qualitative results and future improvements, please refer to chapter 4 of the Executive Summary or chapter 5 of the Complete Thesis linked at the beginning of this README.






