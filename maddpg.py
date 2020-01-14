# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
import torch.nn.functional as F
from utilities import soft_update, transpose_to_tensor, transpose_list, parse_samples
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

# Adapted from Udacity codes
class MADDPG:
    def __init__(self, discount_factor=0.95, tau=0.02, hidden_network=[128, 128], lr_actor=1e-2, lr_critic=1e-2, allow_bn=True, batch_size=128):
        super(MADDPG, self).__init__()

        # DDPGAgent inputs:
        # in_actor = 24
        # out_actor = 2
        # in_critic = obs_full + actions = 24+24+2+2=52
        # out_critic = 1
        self.maddpg_agent = [DDPGAgent(24, hidden_network, 2, 52, hidden_network, lr_actor=lr_actor, lr_critic=lr_critic, allow_bn=allow_bn),
                             DDPGAgent(24, hidden_network, 2, 52, hidden_network, lr_actor=lr_actor, lr_critic=lr_critic, allow_bn=allow_bn)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.batch_size = batch_size

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        obs_all_agents = torch.from_numpy(obs_all_agents).float()
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """
        obs, obs_full, action, reward, next_obs, next_obs_full, done = samples
        
        # get the current agent
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()
        
        # make input to agent
        actor_next_obs = transpose_to_tensor([transpose_list(next_obs)])
        
        # forward the actor network to get the actions
        target_actions = self.target_act(actor_next_obs)
        target_actions = torch.cat(target_actions, dim=0)
        target_actions = torch.cat((target_actions[0], target_actions[1]), dim=1)
        
        next_obs_full = transpose_to_tensor([next_obs_full])[0]
        
        target_critic_input = torch.cat((next_obs_full,target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
            
        reward = transpose_to_tensor(transpose_list(reward))
        
        done = transpose_to_tensor(transpose_list(done))
        
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        
        obs_full = transpose_to_tensor([obs_full])[0]
        
        action = transpose_to_tensor([transpose_list(action)])
        
        # combine all the actions and observations for input to critic
        action = torch.cat(action, dim=0)
        action = torch.cat((action[0], action[1]), dim=1)
        
        # combine all obs with the actions for input to critic
        critic_input = torch.cat((obs_full, action), dim=1).to(device)
        
        # forward the critic network
        q = agent.critic(critic_input)

        # calculate critic loss
        critic_loss = F.mse_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()

        agent.actor_optimizer.zero_grad()
        
        obs = transpose_to_tensor(transpose_list(obs))
        
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]
                
        q_input = torch.cat(q_input, dim=1)
        q_input = torch.cat((obs_full, q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        # actor loss and critic loss, in case we need then in the future
        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)