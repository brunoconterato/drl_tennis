from collections import deque, namedtuple
import random
import torch
import numpy as np
from utilities import transpose_list, transpose_to_tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Adapted from Udacity codes
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "states_full", "actions", "rewards", "next_states", "next_states_full", "dones"])
        self.seed = random.seed(seed)
    
    def add(self, states, states_full, actions, rewards, next_states, next_states_full, dones):
        """Add a new experience to memory."""
        e = self.experience(states, states_full, actions, rewards, next_states, next_states_full, dones)
        
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = [experiences[i].states for i in range(self.batch_size)]
        states_full = np.vstack([e.states_full for e in experiences if e is not None])
        actions = [experiences[i].actions for i in range(self.batch_size)]
        rewards = [experiences[i].rewards for i in range(self.batch_size)]
        next_states = [experiences[i].next_states for i in range(self.batch_size)]
        next_states_full = np.vstack([e.next_states_full for e in experiences if e is not None])
        dones = [experiences[i].dones for i in range(self.batch_size)]
        
        return (states, states_full, actions, rewards, next_states, next_states_full, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

