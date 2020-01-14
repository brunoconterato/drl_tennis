# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Network
from utilities import hard_update, gumbel_softmax, onehot_from_logits, is_numpy
from torch.optim import Adam
import torch
import numpy as np


# add OU noise for exploration
from OUNoise import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

# Adapted from Udacity codes
class DDPGAgent:
    def __init__(self, in_actor, actor_hidden_dimensions, out_actor, in_critic, critic_hidden_dimensions, lr_actor=1.0e-2, lr_critic=1.0e-2, allow_bn=True):
        super(DDPGAgent, self).__init__()

        self.actor = Network(in_actor, actor_hidden_dimensions, out_actor, actor=True).to(device)
        self.critic = Network(in_critic, critic_hidden_dimensions, 1).to(device)
        self.target_actor = Network(in_actor, actor_hidden_dimensions, out_actor, actor=True).to(device)
        self.target_critic = Network(in_critic, critic_hidden_dimensions, 1).to(device)

        self.noise = OUNoise(out_actor, scale=1.0 )
        
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)

    def act(self, obs, noise=0.0):
        if is_numpy(obs):
            obs = torch.from_numpy(obs).float()
        obs = obs.to(device)
            
        action = self.actor(obs) + noise*self.noise.noise()
        action = action.detach().cpu().numpy()
        return torch.from_numpy(np.clip(action, -1, 1))

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs) + noise*self.noise.noise()
        return action
