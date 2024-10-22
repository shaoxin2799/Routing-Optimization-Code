omnet++ code：
// NetworkTopology.ned
network NetworkTopology
{
    submodules:
        nodeA: StandardHost;
        nodeB: StandardHost;
        nodeC: StandardHost;
        nodeD: StandardHost;
        nodeE: StandardHost;
        nodeF: StandardHost;
        nodeG: StandardHost;
        nodeH: StandardHost;
        nodeI: StandardHost;

    connections:
        nodeA.pppg++ <--> { delay = 10ms; } <--> nodeB.pppg++;
        nodeA.pppg++ <--> { delay = 10ms; } <--> nodeD.pppg++;
        nodeA.pppg++ <--> { delay = 10ms; } <--> nodeE.pppg++;
        nodeA.pppg++ <--> { delay = 10ms; } <--> nodeG.pppg++;

        nodeB.pppg++ <--> { delay = 10ms; } <--> nodeC.pppg++;
        nodeD.pppg++ <--> { delay = 10ms; } <--> nodeI.pppg++;
        nodeE.pppg++ <--> { delay = 10ms; } <--> nodeF.pppg++;
        nodeG.pppg++ <--> { delay = 10ms; } <--> nodeH.pppg++;
        nodeC.pppg++ <--> { delay = 10ms; } <--> nodeI.pppg++;
        nodeF.pppg++ <--> { delay = 10ms; } <--> nodeI.pppg++;
        nodeH.pppg++ <--> { delay = 10ms; } <--> nodeI.pppg++;
}
// RoutingAlgorithm.cc
#include "inet/common/INETDefs.h"
#include "inet/networklayer/ipv4/Ipv4RoutingTable.h"
#include "inet/networklayer/ipv4/Ipv4.h"
#include "inet/networklayer/contract/INetfilter.h"
#include "inet/networklayer/routing/IRoutingTable.h"
#include "RoutingAlgorithm.h"

Define_Module(RoutingAlgorithm);

void RoutingAlgorithm::initialize(int stage) {
    if (stage == inet::INITSTAGE_ROUTING_PROTOCOLS) {
        routingTable = getModuleFromPar<inet::Ipv4RoutingTable>(par("routingTableModule"), this);
        scheduleAt(simTime() + par("updateInterval"), new cMessage("update"));
    }
}

void RoutingAlgorithm::handleMessage(cMessage *msg) {
    if (msg->isSelfMessage()) {
        // Call your routing logic here based on the chosen algorithm
        performRoutingUpdate();
        scheduleAt(simTime() + par("updateInterval"), msg);
    } else {
        delete msg;
    }
}

void RoutingAlgorithm::performRoutingUpdate() {
    // Collect network metrics such as delay, packet loss, throughput, etc.
    double currentDelay = measureDelay();
    double currentLoss = measurePacketLoss();
    double currentThroughput = measureThroughput();

    // Depending on the algorithm chosen (SPF, DDPGOR, or TD3OR), route packets accordingly
    if (algorithm == "SPF") {
        // Implement Shortest Path First (SPF) logic here
        performSPF();
    } else if (algorithm == "DDPGOR") {
        // Implement DDPGOR routing logic here
        performDDPGOR();
    } else if (algorithm == "TD3OR") {
        // Implement TD3OR routing logic here
        performTD3OR();
    }
}

void RoutingAlgorithm::performSPF() {
    // SPF logic for routing
}

void RoutingAlgorithm::performDDPGOR() {
    // DDPGOR logic for routing
}

void RoutingAlgorithm::performTD3OR() {
    // TD3OR logic for routing
}

double RoutingAlgorithm::measureDelay() {
    // Logic to measure delay
    return uniform(0.01, 0.1);  // Simulated delay
}

double RoutingAlgorithm::measurePacketLoss() {
    // Logic to measure packet loss
    return uniform(0.0, 0.05);  // Simulated packet loss
}

double RoutingAlgorithm::measureThroughput() {
    // Logic to measure throughput
    return uniform(10.0, 100.0);  // Simulated throughput
}
// RoutingAlgorithm.h
#ifndef __INET_ROUTINGALGORITHM_H
#define __INET_ROUTINGALGORITHM_H

#include "inet/common/INETDefs.h"
#include "inet/networklayer/ipv4/Ipv4RoutingTable.h"
#include "inet/networklayer/ipv4/Ipv4.h"

class RoutingAlgorithm : public cSimpleModule {
  protected:
    inet::Ipv4RoutingTable *routingTable = nullptr;

    virtual void initialize(int stage) override;
    virtual void handleMessage(cMessage *msg) override;

    // Routing logic
    void performRoutingUpdate();

    // SPF Routing
    void performSPF();
    
    // DDPGOR Routing
    void performDDPGOR();

    // TD3OR Routing
    void performTD3OR();

    // Helper functions to measure network metrics
    double measureDelay();
    double measurePacketLoss();
    double measureThroughput();
    
    // Algorithm selection
    std::string algorithm;
};

#endif
[General]
network = NetworkTopology
**.updateInterval = 1s  # Time interval to update routing decisions

# Parameters for different routing algorithms
**.algorithm = "SPF"  # Change this to "DDPGOR" or "TD3OR" to switch algorithms

# Simulation limits
sim-time-limit = 100s
output-scalar-file = "routing-results.sca"
import matplotlib.pyplot as plt
import pandas as pd

# Example data from simulation (to be replaced with actual data from OMNeT++)
data = {
    "episode": range(1, 121),
    "DDPGOR_reward": [4 + i * 0.05 for i in range(120)],
    "TD3OR_reward": [5 + i * 0.06 for i in range(120)],
    "SPF_delay": [2000 - i * 10 for i in range(100)],
    "DDPGOR_delay": [1000 - i * 5 for i in range(100)],
    "TD3OR_delay": [500 - i * 3 for i in range(100)],
}

df = pd.DataFrame(data)

# Plot Reward vs Episode for DDPGOR and TD3OR
plt.figure(figsize=(8,6))
plt.plot(df["episode"], df["DDPGOR_reward"], label="DDPGOR", color="black")
plt.plot(df["episode"], df["TD3OR_reward"], label="TD3OR", color="red")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.title("Reward vs Episode (DDPGOR vs TD3OR)")
plt.grid(True)
plt.show()

# Plot Time Delay vs Episode for SPF, DDPGOR, and TD3OR
plt.figure(figsize=(8,6))
plt.plot(df["episode"][:100], df["SPF_delay"], label="SPF", color="blue")
plt.plot(df["episode"][:100], df["DDPGOR_delay"], label="DDPGOR", color="red")
plt.plot(df["episode"][:100], df["TD3OR_delay"], label="TD3OR", color="green")
plt.xlabel("Episode")
plt.ylabel("Time Delay (ms)")
plt.legend()
plt.title("Time Delay vs Episode (SPF vs DDPGOR vs TD3OR)")
plt.grid(True)
plt.show()

# Example bar plot for delay vs traffic intensity
traffic_intensity = ['20%', '40%', '60%', '80%', '100%', '120%']
spf_delay = [1500, 1800, 2200, 2500, 3000, 3500]
ddpgor_delay = [1300, 1500, 1800, 2000, 2300, 2500]
td3or_delay = [1200, 1400, 1600, 1800, 2000, 2200]

plt.figure(figsize=(8,6))
plt.bar(traffic_intensity, spf_delay, label="SPF", color="blue")
plt.bar(traffic_intensity, ddpgor_delay, label="DDPGOR", color="orange", alpha

PYthon code：
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TD3ORModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TD3ORModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = self.fc3(x)
        return action

# Training TD3OR
state_dim = 3  # delay, packet loss, throughput
action_dim = 1  # next-hop decision
model = TD3ORModel(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (example)
for episode in range(1000):
    state = np.random.rand(state_dim)  # Simulate network state
    state_tensor = torch.FloatTensor(state)
    action = model(state_tensor)
    
    # Calculate reward based on action (example reward)
    reward = -np.sum(np.square(action.detach().numpy()))  # Reward function example
    
    # Backpropagation
    optimizer.zero_grad()
    loss = -reward  # Minimise negative reward
    loss.backward()
    optimizer.step()

# Save the model
torch.jit.save(torch.jit.script(model), "td3or_model.pt")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.layer1(state))
        a = torch.relu(self.layer2(a))
        a = torch.tanh(self.layer3(a)) * self.max_action
        return a

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, state, action):
        q = torch.relu(self.layer1(torch.cat([state, action], 1)))
        q = torch.relu(self.layer2(q))
        q = self.layer3(q)
        return q

# DDPGOR Algorithm
class DDPGOR(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):
        for it in range(iterations):
            # Sample a batch of transitions from replay buffer
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            reward = torch.FloatTensor(reward).to(device)
            not_done = torch.FloatTensor(not_done).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (not_done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = nn.MSELoss()(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# TD3OR Algorithm
class TD3OR(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1 = Critic(state_dim, action_dim).to(device)
        self.critic_target_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=3e-4)

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, noise_std=0.2, noise_clip=0.5):
        for it in range(iterations):
            # Sample a batch of transitions from replay buffer
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            reward = torch.FloatTensor(reward).to(device)
            not_done = torch.FloatTensor(not_done).to(device)

            # Select next action according to target policy and add clipped noise
            noise = (torch.randn_like(action) * noise_std).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1 = self.critic_target_1(next_state, next_action)
            target_Q2 = self.critic_target_2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1 = self.critic_1(state, action)
            current_Q2 = self.critic_2(state, action)

            # Compute critic loss
            critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % 2 == 0:
                # Compute actor loss
                actor_loss = -self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Replay Buffer
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state, action, next_state, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))

        return np.array(batch_states), np.array(batch_actions), np.array(batch_next_states), np.array(batch_rewards), np.array(batch_dones)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment Setup
if __name__ == "__main__":
    # Environment parameters
    state_dim = 14  # Example state dimension (e.g., number of nodes in a network)
    action_dim = 4  # Example action dimension (e.g., number of possible routes)
    max_action = 1.0

    # Initialize DDPGOR and TD3OR agents
    ddpgor_agent = DDPGOR(state_dim, action_dim, max_action)
    td3or_agent = TD3OR(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    # Training parameters
    total_episodes = 120  # Number of episodes
    episode_length = 100  # Length of each episode
    batch_size = 64

    # Simulate training data and train agents
    rewards_ddpgor = []
    rewards_td3or = []
    for episode in range(total_episodes):
        state = np.random.randn(state_dim)  # Initialize state randomly
        episode_reward_ddpgor = 0
        episode_reward_td3or = 0
        for t in range(episode_length):
            action_ddpgor = ddpgor_agent.select_action(state)
            action_td3or = td3or_agent.select_action(state)
            next_state = np.random.randn(state_dim)  # Simulate next state
            reward_ddpgor = np.random.uniform(-1, 1)  # Simulate reward
            reward_td3or = np.random.uniform(-1, 1)  # Simulate reward
            done = random.choice([0, 1])  # Randomly decide if episode ends

            replay_buffer.add((state, action_ddpgor, next_state, reward_ddpgor, 1 - done))
            state = next_state
            episode_reward_ddpgor += reward_ddpgor
            episode_reward_td3or += reward_td3or

            # Train the agents
            if len(replay_buffer.storage) > batch_size:
                ddpgor_agent.train(replay_buffer, 1, batch_size)
                td3or_agent.train(replay_buffer, 1, batch_size)

        rewards_ddpgor.append(episode_reward_ddpgor)
        rewards_td3or.append(episode_reward_td3or)

    # Plotting reward over episodes
