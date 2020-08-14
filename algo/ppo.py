import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np


class PPO(object):
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 a_lr=None,
                 c_lr=None,
                 eps=None,
                 max_grad_norm=None,
                 comm=None):

        self.actor_critic = actor_critic
        self.cp_actor_critic = copy.deepcopy(actor_critic)

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.a_lr = a_lr
        self.c_lr = c_lr
        self.eps = eps

        # self.optimizer = optim.Adam([{'params': actor_critic.actor.parameters(), 'lr': a_lr},
        #                              {'params': actor_critic.critic.parameters(), 'lr': c_lr},
        #                              {'params': actor_critic.critic_linear.parameters(), 'lr': c_lr},
        #                              {'params': actor_critic.dist.parameters(), 'lr': a_lr}],
        #                             lr=a_lr,
        #                             eps=eps,
        #                             weight_decay=1e-6)

        self.optimizer = optim.Adam([{'params': actor_critic.actor.parameters(), 'lr': a_lr},
                                     {'params': actor_critic.critic.parameters(), 'lr': c_lr},
                                     {'params': actor_critic.critic_linear.parameters(), 'lr': c_lr},
                                     {'params': actor_critic.dist.parameters(), 'lr': a_lr}],
                                    lr=a_lr,
                                    eps=eps)
        # self.optimizer = optim.Adam(actor_critic.parameters(), lr=a_lr, eps=eps, weight_decay=1e-6)

        # TODO,
        self.comm = comm
        self.grad_list = []
        self.cum_indices = []
        self.send_data = np.array([], dtype='float32')

        for param in self.cp_actor_critic.parameters():
            self.cum_indices.append(param.data.nelement())
        self.cum_indices = np.cumsum(self.cum_indices)


        # TODO, additional variables
        self.advantages = None
        self.data_generator = None
        self.rollouts = None

    def update(self, rollouts):
        # discounted return 에서 value(predicted)를 빼준다.
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]  # 맨 마지막 값은 제외하고 계산.
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)    # normalize

        for e in range(self.ppo_epoch):
            if hasattr(self.actor_critic, 'gru'):
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            elif hasattr(self.actor_critic, 'seq'):
                data_generator = rollouts.sequence_generator(
                    advantages, self.num_mini_batch)
            else:
                # random set 데이터 생성, rollout stack에서 하나의 mini_batch_size의 데이터를 추출한다.
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            # mini_batch_size 만큼 for loop돈다.
            for sample in data_generator:
                observations_batch, states_batch, actions_batch, \
                   return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
                    observations_batch, states_batch,
                    masks_batch, actions_batch)
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()  # TODO

                # TODO, detaching hidden states of LSTM
                self.actor_critic.init_hidden()

                # backward()를 호출하면 중간에 저장된 Tensor 값들이 없어진다.
                # retain_graph=True를 옵션으로 주면 중간 값들을 없애지 않고 저장함
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()  # calc gradient
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()       # actual parameter update

        return value_loss, action_loss, dist_entropy

