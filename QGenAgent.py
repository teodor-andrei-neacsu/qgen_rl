"""
RL agent that uses DQlearning to select possible answers to quesitons. 


Action:
State:
Reward: 

"""
import torch
import torch.nn as nn

class Qgen_DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Qgen_DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):


        action = "some selected span"
        
        return action


class QGen_Agent()
