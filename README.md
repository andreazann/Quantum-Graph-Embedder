# Single-Agent Graph Embedding for Quantum Computers via Reinforcement Learning - Master Thesis @ Politecnico di Milano

pytorch > gym > stablebaselines3 

Full thesis at [Thesis_Complete.pdf](https://github.com/user-attachments/files/17742457/Thesis_Complete.pdf)

Thesis summary at [Thesis_Executive_Summary.pdf](https://github.com/user-attachments/files/17742460/Thesis_Executive_Summary.pdf)

This project discusses how supercomputers alone are insufficient for tackling NP-hard problems, exemplified by the Graph Minor Embedding problem, crucial for quantum computing with D-Wave systems. 
The work focuses on using Reinforcement Learning (RL) techniques to enhance heuristics like D-Wave's minorminer for embedding problems on a Quantum Processing Unit (QPU). 
The study explores PPO and DQN RL models to optimize embeddings, aiming to reduce the number of qubits needed. Performance comparisons are made against the state-of-the-art, 
with evaluations on datasets of 15, 30, and 50-node graphs.

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
For this reason the reward function has been defined as: log(*prevAvgHeat/currAvgHeat*)
Where *prevAvgHeat* represents the average heat of the graph at timestep - 1, while currAvgHeat represents the average heat of the graph at the current timestep.

## 2.7 Episode termination condition

As stated in 2.6 the reward function is expressed as a progress at each timestep of the difference of heat with respect to the previous timestep, it would be convenient to base the episode termination
condition on the same concept. In fact if we set this condition as a percentage of heat to reach we will obtain a total reward for each episode that is very similar between graphs having
different complexities. Given that we have a starting average heat defined as startAvgHeat, a current average heat as currAvgHeat and a percentage of progress of i.e. 10% passed as parameter
to the agent, the episode termination condition will be defined as: *currAvgHeat* ≤ *startAvgHeat* ∗ (1 − 0.1)

## 2.8 Definition of Single-Traveler agent on a graph

The concept proposed consists of having a **single agent** able only to view a **neighborhood of nodes** at each timestep. Hopping between neighborhoods where the heat higher, lowering
it using the **Node Expansion** technique and processing a new neighborhood on the next timestep. In this way the agent will have a limited view of the graph, keeping the elements
of the observation space contained while updating the parameters on the same agent improving it at each timestep. The following pseudocode shows the general behavior of the RL agent,
identified in a big portion of the step function of the Custom Gym Environment:

<img width="607" alt="single-agent-algorithm" src="https://github.com/user-attachments/assets/9d484344-2200-4e7f-a871-5c516105fc77">

continuing readme today.. 14/11/2024








