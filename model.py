import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from distributions import get_distribution


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        hidden_critic, hidden_actor, states = self(inputs, states, masks)

        action, action_mean, action_std = self.dist.sample(hidden_actor[-1], deterministic=deterministic)    # action을 mean과 std의 형태로 sampling해서 구함
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(hidden_actor[-1], action)

        value = self.critic_linear(hidden_critic[-1])

        return value, action, action_log_probs, states, hidden_actor, hidden_critic

    # actor -> and then critic for Q-function estimation
    def actQ(self, inputs, states, masks, deterministic=False):
        hidden_actor = self(inputs, states, masks, target='actor')

        action, action_mean, action_std = self.dist.sample(hidden_actor[-1], deterministic=deterministic)    # action을 mean과 std의 형태로 sampling해서 구함
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(hidden_actor[-1], action)

        states.flatten()[-7:] = action.flatten()
        states = states.reshape(1, -1)
        hidden_critic = self(inputs, states, masks, target='critic')
        value = self.critic_linear(hidden_critic[-1])

        return value, action, action_log_probs, states, hidden_actor, hidden_critic

    def get_value(self, inputs, states, masks):        
        hidden_critic, _, states = self(inputs, states, masks)
        value = self.critic_linear(hidden_critic[-1])
        return value

    def get_valueQ(self, inputs, states, masks, deterministic=False):
        hidden_actor = self(inputs, states, masks, target='actor')
        action, action_mean, action_std = self.dist.sample(hidden_actor[-1], deterministic=deterministic)
        states.flatten()[-7:] = action.flatten()
        states = states.reshape(1, -1)
        hidden_critic = self(inputs, states, masks, target='critic')
        value = self.critic_linear(hidden_critic[-1])
        return value, states
    
    def evaluate_actions(self, inputs, states, masks, actions):
        hidden_critic, hidden_actor, states = self(inputs, states, masks)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(hidden_actor[-1], actions)

        value = self.critic_linear(hidden_critic[-1])

        return value, action_log_probs, dist_entropy, states

    def evaluate_actionsQ(self, inputs, states, masks, actions):     # input: observation, [num_step x 11]
        hidden_actor = self(inputs, states, masks, target='actor')
        hidden_critic = self(inputs, states, masks, target='critic')
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(hidden_actor[-1], actions)

        value = self.critic_linear(hidden_critic[-1])

        return value, action_log_probs, dist_entropy, states


class CONVPolicy(Policy):
    def __init__(self, num_inputs, action_space):
        super(CONVPolicy, self).__init__()

        # image size: (180, 180)
        self.main = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(16 * 7 * 7, 512),
            nn.ReLU()
        )

        self.critic_linear = nn.Linear(512, 1)

        self.dist = get_distribution(512, action_space)

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)

        def mult_gain(m):
            relu_gain = nn.init.calculate_gain('relu')
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                m.weight.data.mul_(relu_gain)
    
        self.main.apply(mult_gain)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        x = self.main(inputs)

        return x, x, states     # x: cricit, actor, states


class CNNPolicy(Policy):
    def __init__(self, num_inputs, action_space, use_gru):
        super(CNNPolicy, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU()
        )

        if use_gru:
            self.gru = nn.GRUCell(512, 512)

        self.critic_linear = nn.Linear(512, 1)

        self.dist = get_distribution(512, action_space)

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    def reset_parameters(self):
        self.apply(weights_init)

        def mult_gain(m):
            relu_gain = nn.init.calculate_gain('relu')
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                m.weight.data.mul_(relu_gain)

        self.main.apply(mult_gain)

        if hasattr(self, 'gru'):
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        x = self.main(inputs / 255.0)

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)
        return x, x, states


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class RNNPolicy(Policy):
    def __init__(self, num_actor_inputs, num_critic_inputs, action_space, use_gru=False, cuda_use=False):
        super(RNNPolicy, self).__init__()

        self.action_space = action_space
        self.nNode = 64
        self.hidden_dim = 64
        self.cuda_use = cuda_use

        if use_gru == True:
            self.gru = 0

        self.actor = nn.Sequential(
            nn.Linear(num_actor_inputs, self.nNode),
            nn.Tanh()
            # nn.ReLU()
        )

        self.actor_lstm = nn.LSTM(self.nNode, self.hidden_dim, num_layers=1)
        self.a_lstm_hidden = self.init_hidden()

        self.critic = nn.Sequential(
            nn.Linear(num_critic_inputs, self.nNode),
            nn.Tanh()
            # nn.ReLU()
        )

        self.critic_lstm = nn.LSTM(self.nNode, self.hidden_dim, num_layers=1)
        self.c_lstm_hidden = self.init_hidden()

        self.critic_linear = nn.Linear(self.hidden_dim, 1)
        self.dist = get_distribution(self.hidden_dim, action_space)

        self.train()
        self.reset_parameters()

    def init_hidden(self):
        # (num_layers, batch, hidden_size)
        if self.cuda_use:
            h = torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda()
            c = torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda()
        else:
            h = torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
            c = torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim))

        self.c_lstm_hidden = (h, c)
        self.a_lstm_hidden = (h, c)


    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init_mlp)
        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):  # input은 observation값
        rst_a = self.actor(inputs)
        self.actor_lstm.flatten_parameters()
        a_lstm_out, self.a_lstm_hidden = self.actor_lstm(rst_a.view(len(inputs), 1, -1), self.a_lstm_hidden)

        rst_c = self.critic(inputs)
        self.critic_lstm.flatten_parameters()
        c_lstm_out, self.c_lstm_hidden = self.critic_lstm(rst_c.view(len(inputs), 1, -1), self.c_lstm_hidden)

        # hidden_actor_list = [self.actor_linear_final(a_lstm_out.view(len(inputs), -1))]
        # hidden_critic_list = [self.critic_linear_final(c_lstm_out.view(len(inputs), -1))]
        hidden_actor_list = [a_lstm_out.view(len(inputs), -1)]
        hidden_critic_list = [c_lstm_out.view(len(inputs), -1)]

        return hidden_critic_list, hidden_actor_list, states


class MLPPolicy(Policy):
    def __init__(self, num_actor_inputs, num_critic_inputs, action_space, symm_policy=True, use_seq=False, cuda_use=False):
        super(MLPPolicy, self).__init__()

        self.action_space = action_space
        self.nNode = 512         # 64, 128
        self.hidden_dim = 512    # 64, 128
        self.cuda_use = cuda_use
        self.symm_policy = symm_policy

        if use_seq == True:
            self.seq = 0

        # as input, (N, Cin, L), N: batch size, Cin: input size, L: length of signal seq
        self.actor = nn.Sequential(
            nn.Linear(num_actor_inputs, self.nNode),

            # nn.Tanh(),
            nn.ReLU(),

            nn.Linear(self.nNode, self.hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),

            nn.Linear(self.nNode, self.hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),
        )

        self.critic = nn.Sequential(
            nn.Linear(num_critic_inputs, self.nNode),
            # nn.Tanh(),
            nn.ReLU(),

            nn.Linear(self.nNode, self.hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),

            nn.Linear(self.nNode, self.hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),
        )

        self.critic_linear = nn.Linear(self.hidden_dim, 1)  # self.hidden_dim
        self.dist = get_distribution(self.hidden_dim, action_space)  # self.hidden_dim

        self.train()
        self.reset_parameters()

    def init_hidden(self):
        # do nothing in MLP class.
        pass

    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init_mlp)
        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        # a_inputs = inputs[:, :, :-1].view(inputs.size(0), -1)
        a_inputs = inputs.view(inputs.size(0), -1)
        if self.symm_policy:
            c_inputs = inputs.view(inputs.size(0), -1)
        else:
            c_inputs = states.view(states.size(0), -1)

        hidden_actor_list = [self.actor(a_inputs)]
        hidden_critic_list = [self.critic(c_inputs)]

        return hidden_critic_list, hidden_actor_list, states

    # def forward(self, inputs, states, masks, target):
    #     if target == 'actor':
    #         a_inputs = inputs.view(inputs.size(0), -1)
    #         hidden_actor_list = [self.actor(a_inputs)]
    #         return hidden_actor_list
    #     elif target == 'critic':
    #         if self.symm_policy:
    #             c_inputs = inputs.view(inputs.size(0), -1)
    #         else:
    #             c_inputs = states.view(states.size(0), -1)
    #         hidden_critic_list = [self.critic(c_inputs)]
    #         return hidden_critic_list
    #     else:
    #         raise ValueError