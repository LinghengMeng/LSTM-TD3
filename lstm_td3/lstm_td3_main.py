from copy import deepcopy
import numpy as np
# import pybulletgym      # To register tasks in PyBulletGym
import pybullet_envs    # To register tasks in PyBullet
import gym
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from lstm_td3.utils.logx import EpochLogger, setup_logger_kwargs, colorize
import itertools
from lstm_td3.env_wrapper.pomdp_wrapper import POMDPWrapper
from lstm_td3.env_wrapper.env import make_bullet_task
import os
import os.path as osp
import json
from collections import namedtuple


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for agents.
    """

    def __init__(self, obs_dim, act_dim, max_size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = max_size
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = list(next_obs)
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def sample_batch_with_history(self, batch_size=32, max_hist_len=100):
        """

        :param batch_size:
        :param max_hist_len: the length of experiences before current experience
        :return:
        """
        idxs = np.random.randint(max_hist_len, self.size, size=batch_size)
        # History
        if max_hist_len == 0:
            hist_obs = np.zeros([batch_size, 1, self.obs_dim])
            hist_act = np.zeros([batch_size, 1, self.act_dim])
            hist_obs2 = np.zeros([batch_size, 1, self.obs_dim])
            hist_act2 = np.zeros([batch_size, 1, self.act_dim])
            hist_rew = np.zeros([batch_size, 1])
            hist_done = np.zeros([batch_size, 1])
            hist_len = np.zeros(batch_size)
        else:
            hist_obs = np.zeros([batch_size, max_hist_len, self.obs_dim])
            hist_act = np.zeros([batch_size, max_hist_len, self.act_dim])
            hist_obs_len = max_hist_len * np.ones(batch_size)
            hist_obs2 = np.zeros([batch_size, max_hist_len, self.obs_dim])
            hist_act2 = np.zeros([batch_size, max_hist_len, self.act_dim])
            hist_obs2_len = max_hist_len * np.ones(batch_size)

            # Extract history experiences before sampled index
            for i, id in enumerate(idxs):
                hist_start_id = id - max_hist_len
                if hist_start_id < 0:
                    hist_start_id = 0
                # If exist done before the last experience (not including the done in id), start from the index next to the done.
                if len(np.where(self.done_buf[hist_start_id:id] == 1)[0]) != 0:
                    hist_start_id = hist_start_id + (np.where(self.done_buf[hist_start_id:id] == 1)[0][-1]) + 1
                hist_seg_len = id - hist_start_id
                hist_obs_len[i] = hist_seg_len
                hist_obs[i] = self.obs_buf[hist_start_id:id]
                hist_act[i] = self.act_buf[hist_start_id:id]
                if hist_seg_len == 0:
                    hist_obs2_len[i] = 1
                else:
                    hist_obs2_len[i] = hist_seg_len
                hist_obs2[i] = self.obs2_buf[hist_start_id:id]
                hist_act2[i] = self.act_buf[hist_start_id+1:id+1]

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     hist_obs=hist_obs,
                     hist_act=hist_act,
                     hist_obs2=hist_obs2,
                     hist_act2=hist_act2,
                     hist_obs_len=hist_obs_len,
                     hist_obs2_len=hist_obs2_len)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
#######################################################################################

#######################################################################################


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 mem_pre_lstm_hid_sizes=(128,),
                 mem_lstm_hid_sizes=(128,),
                 mem_after_lstm_hid_size=(128,),
                 cur_feature_hid_sizes=(128,),
                 post_comb_hid_sizes=(128,),
                 hist_with_past_act=False):
        super(MLPCritic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hist_with_past_act = hist_with_past_act
        #
        self.mem_pre_lstm_layers = nn.ModuleList()
        self.mem_lstm_layers = nn.ModuleList()
        self.mem_after_lstm_layers = nn.ModuleList()

        self.cur_feature_layers = nn.ModuleList()
        self.post_combined_layers = nn.ModuleList()
        # Memory
        #    Pre-LSTM
        if self.hist_with_past_act:
            mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
        else:
            mem_pre_lstm_layer_size = [obs_dim] + list(mem_pre_lstm_hid_sizes)
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                   mem_pre_lstm_layer_size[h + 1]),
                                         nn.ReLU()]
        #    LSTM
        self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)
        for h in range(len(self.mem_lstm_layer_sizes) - 1):
            self.mem_lstm_layers += [
                nn.LSTM(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1], batch_first=True)]

        #   After-LSTM
        self.mem_after_lstm_layer_size = [self.mem_lstm_layer_sizes[-1]] + list(mem_after_lstm_hid_size)
        for h in range(len(self.mem_after_lstm_layer_size)-1):
            self.mem_after_lstm_layers += [nn.Linear(self.mem_after_lstm_layer_size[h],
                                                     self.mem_after_lstm_layer_size[h+1]),
                                           nn.ReLU()]

        # Current Feature Extraction
        cur_feature_layer_size = [obs_dim + act_dim] + list(cur_feature_hid_sizes)
        for h in range(len(cur_feature_layer_size) - 1):
            self.cur_feature_layers += [nn.Linear(cur_feature_layer_size[h], cur_feature_layer_size[h + 1]),
                                        nn.ReLU()]

        # Post-Combination
        post_combined_layer_size = [self.mem_after_lstm_layer_size[-1] + cur_feature_layer_size[-1]] + list(
            post_comb_hid_sizes) + [1]
        for h in range(len(post_combined_layer_size) - 2):
            self.post_combined_layers += [nn.Linear(post_combined_layer_size[h], post_combined_layer_size[h + 1]),
                                          nn.ReLU()]
        self.post_combined_layers += [nn.Linear(post_combined_layer_size[-2], post_combined_layer_size[-1]),
                                      nn.Identity()]

    def forward(self, obs, act, hist_obs, hist_act, hist_seg_len):
        #
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        if self.hist_with_past_act:
            x = torch.cat([hist_obs, hist_act], dim=-1)
        else:
            x = hist_obs
        # Memory
        #    Pre-LSTM
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)
        #    LSTM
        for layer in self.mem_lstm_layers:
            x, (lstm_hidden_state, lstm_cell_state) = layer(x)
        #    After-LSTM
        for layer in self.mem_after_lstm_layers:
            x = layer(x)
        #    History output mask to reduce disturbance cased by none history memory
        hist_out = torch.gather(x, 1,
                                (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.mem_after_lstm_layer_size[-1]).unsqueeze(
                                    1).long()).squeeze(1)

        # Current Feature Extraction
        x = torch.cat([obs, act], dim=-1)
        for layer in self.cur_feature_layers:
            x = layer(x)

        # Post-Combination
        extracted_memory = hist_out
        x = torch.cat([extracted_memory, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)
        # squeeze(x, -1) : critical to ensure q has right shape.
        return torch.squeeze(x, -1), extracted_memory


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit,
                 mem_pre_lstm_hid_sizes=(128,),
                 mem_lstm_hid_sizes=(128,),
                 mem_after_lstm_hid_size=(128,),
                 cur_feature_hid_sizes=(128,),
                 post_comb_hid_sizes=(128,),
                 hist_with_past_act=False):
        super(MLPActor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.hist_with_past_act = hist_with_past_act
        #
        self.mem_pre_lstm_layers = nn.ModuleList()
        self.mem_lstm_layers = nn.ModuleList()
        self.mem_after_lstm_layers = nn.ModuleList()

        self.cur_feature_layers = nn.ModuleList()
        self.post_combined_layers = nn.ModuleList()

        # Memory
        #    Pre-LSTM
        if self.hist_with_past_act:
            mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
        else:
            mem_pre_lstm_layer_size = [obs_dim] + list(mem_pre_lstm_hid_sizes)
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                   mem_pre_lstm_layer_size[h + 1]),
                                         nn.ReLU()]
        #    LSTM
        self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)
        for h in range(len(self.mem_lstm_layer_sizes) - 1):
            self.mem_lstm_layers += [
                nn.LSTM(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1], batch_first=True)]
        #   After-LSTM
        self.mem_after_lstm_layer_size = [self.mem_lstm_layer_sizes[-1]] + list(mem_after_lstm_hid_size)
        for h in range(len(self.mem_after_lstm_layer_size) - 1):
            self.mem_after_lstm_layers += [nn.Linear(self.mem_after_lstm_layer_size[h],
                                                     self.mem_after_lstm_layer_size[h + 1]),
                                           nn.ReLU()]

        # Current Feature Extraction
        cur_feature_layer_size = [obs_dim] + list(cur_feature_hid_sizes)
        for h in range(len(cur_feature_layer_size) - 1):
            self.cur_feature_layers += [nn.Linear(cur_feature_layer_size[h], cur_feature_layer_size[h + 1]),
                                        nn.ReLU()]

        # Post-Combination
        post_combined_layer_size = [self.mem_after_lstm_layer_size[-1] + cur_feature_layer_size[-1]] + list(
            post_comb_hid_sizes) + [act_dim]
        for h in range(len(post_combined_layer_size) - 2):
            self.post_combined_layers += [nn.Linear(post_combined_layer_size[h], post_combined_layer_size[h + 1]),
                                          nn.ReLU()]
        self.post_combined_layers += [nn.Linear(post_combined_layer_size[-2], post_combined_layer_size[-1]), nn.Tanh()]

    def forward(self, obs, hist_obs, hist_act, hist_seg_len):
        #
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        if self.hist_with_past_act:
            x = torch.cat([hist_obs, hist_act], dim=-1)
        else:
            x = hist_obs
        # Memory
        #    Pre-LSTM
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)
        #    LSTM
        for layer in self.mem_lstm_layers:
            x, (lstm_hidden_state, lstm_cell_state) = layer(x)
        #    After-LSTM
        for layer in self.mem_after_lstm_layers:
            x = layer(x)
        hist_out = torch.gather(x, 1,
                                (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.mem_after_lstm_layer_size[-1]).unsqueeze(
                                    1).long()).squeeze(1)

        # Current Feature Extraction
        x = obs
        for layer in self.cur_feature_layers:
            x = layer(x)

        # Post-Combination
        extracted_memory = hist_out
        x = torch.cat([extracted_memory, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)
        return self.act_limit * x, extracted_memory


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit=1,
                 critic_mem_pre_lstm_hid_sizes=(128,),
                 critic_mem_lstm_hid_sizes=(128,),
                 critic_mem_after_lstm_hid_size=(128,),
                 critic_cur_feature_hid_sizes=(128,),
                 critic_post_comb_hid_sizes=(128,),
                 critic_hist_with_past_act=False,
                 actor_mem_pre_lstm_hid_sizes=(128,),
                 actor_mem_lstm_hid_sizes=(128,),
                 actor_mem_after_lstm_hid_size=(128,),
                 actor_cur_feature_hid_sizes=(128,),
                 actor_post_comb_hid_sizes=(128,),
                 actor_hist_with_past_act=False):
        super(MLPActorCritic, self).__init__()
        self.q1 = MLPCritic(obs_dim, act_dim,
                            mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                            mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                            mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
                            cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                            post_comb_hid_sizes=critic_post_comb_hid_sizes,
                            hist_with_past_act=critic_hist_with_past_act)
        self.q2 = MLPCritic(obs_dim, act_dim,
                            mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                            mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                            mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
                            cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                            post_comb_hid_sizes=critic_post_comb_hid_sizes,
                            hist_with_past_act=critic_hist_with_past_act)
        self.pi = MLPActor(obs_dim, act_dim, act_limit,
                           mem_pre_lstm_hid_sizes=actor_mem_pre_lstm_hid_sizes,
                           mem_lstm_hid_sizes=actor_mem_lstm_hid_sizes,
                           mem_after_lstm_hid_size=actor_mem_after_lstm_hid_size,
                           cur_feature_hid_sizes=actor_cur_feature_hid_sizes,
                           post_comb_hid_sizes=actor_post_comb_hid_sizes,
                           hist_with_past_act=actor_hist_with_past_act)

    def act(self, obs, hist_obs=None, hist_act=None, hist_seg_len=None, device=None):
        if (hist_obs is None) or (hist_act is None) or (hist_seg_len is None):
            hist_obs = torch.zeros(1, 1, self.obs_dim).to(device)
            hist_act = torch.zeros(1, 1, self.act_dim).to(device)
            hist_seg_len = torch.zeros(1).to(device)
        with torch.no_grad():
            act, _, = self.pi(obs, hist_obs, hist_act, hist_seg_len)
            return act.cpu().numpy()


#######################################################################################

#######################################################################################
def lstm_td3(resume_exp_dir=None,
             env_name='', seed=0,
             steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
             polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
             start_steps=10000,
             update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
             noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
             batch_size=100,
             max_hist_len=100,
             partially_observable=False,
             pomdp_type = 'remove_velocity',
             flicker_prob=0.2, random_noise_sigma=0.1, random_sensor_missing_prob=0.1,
             use_double_critic = True,
             use_target_policy_smooth = True,
             critic_mem_pre_lstm_hid_sizes=(128,),
             critic_mem_lstm_hid_sizes=(128,),
             critic_mem_after_lstm_hid_size=(128,),
             critic_cur_feature_hid_sizes=(128,),
             critic_post_comb_hid_sizes=(128,),
             critic_hist_with_past_act=False,
             actor_mem_pre_lstm_hid_sizes=(128,),
             actor_mem_lstm_hid_sizes=(128,),
             actor_mem_after_lstm_hid_size=(128,),
             actor_cur_feature_hid_sizes=(128,),
             actor_post_comb_hid_sizes=(128,),
             actor_hist_with_past_act=False,
             logger_kwargs=dict(), save_freq=1):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target
            policy.

        noise_clip (float): Limit for absolute value of target policy
            smoothing noise.

        policy_delay (int): Policy will only be updated once every
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    # If not going to resume, create new logger.
    if resume_exp_dir is None:
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())
    else:
        logger = EpochLogger(**logger_kwargs)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Wrapper environment if using POMDP
    if partially_observable:
        env = POMDPWrapper(env_name, pomdp_type, flicker_prob, random_noise_sigma, random_sensor_missing_prob)
        test_env = POMDPWrapper(env_name, pomdp_type, flicker_prob, random_noise_sigma, random_sensor_missing_prob)
    else:
        # env, test_env = gym.make(env_name), gym.make(env_name)
        env = make_bullet_task(env_name, dp_type='MDP')
        test_env = make_bullet_task(env_name, dp_type='MDP')
        env.seed(seed)
        test_env.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = MLPActorCritic(obs_dim, act_dim, act_limit,
                        critic_mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                        critic_mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                        critic_mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
                        critic_cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                        critic_post_comb_hid_sizes=critic_post_comb_hid_sizes,
                        critic_hist_with_past_act=critic_hist_with_past_act,
                        actor_mem_pre_lstm_hid_sizes=actor_mem_pre_lstm_hid_sizes,
                        actor_mem_lstm_hid_sizes=actor_mem_lstm_hid_sizes,
                        actor_mem_after_lstm_hid_size=actor_mem_after_lstm_hid_size,
                        actor_cur_feature_hid_sizes=actor_cur_feature_hid_sizes,
                        actor_post_comb_hid_sizes=actor_post_comb_hid_sizes,
                        actor_hist_with_past_act=actor_hist_with_past_act)
    ac_targ = deepcopy(ac)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ac.to(device)
    ac_targ.to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, max_size=replay_size)

    # # Count variables (protip: try to get a feel for how different size networks behave!)
    # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    # logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        h_o, h_a, h_o2, h_a2, h_o_len, h_o2_len = data['hist_obs'], data['hist_act'], data['hist_obs2'], data['hist_act2'], data['hist_obs_len'], data['hist_obs2_len']

        q1, q1_extracted_memory = ac.q1(o, a, h_o, h_a, h_o_len)
        q2, q2_extracted_memory = ac.q2(o, a, h_o, h_a, h_o_len)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ, _ = ac_targ.pi(o2, h_o2, h_a2, h_o2_len)

            # Target policy smoothing
            if use_target_policy_smooth:
                epsilon = torch.randn_like(pi_targ) * target_noise
                epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
                a2 = pi_targ + epsilon
                a2 = torch.clamp(a2, -act_limit, act_limit)
            else:
                a2 = pi_targ

            # Target Q-values
            q1_pi_targ, _ = ac_targ.q1(o2, a2, h_o2, h_a2, h_o2_len)
            q2_pi_targ, _ = ac_targ.q2(o2, a2, h_o2, h_a2, h_o2_len)

            if use_double_critic:
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            else:
                q_pi_targ = q1_pi_targ
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()

        if use_double_critic:
            loss_q = loss_q1 + loss_q2
        else:
            loss_q = loss_q1

        # Useful info for logging
        # import pdb; pdb.set_trace()
        loss_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                         Q2Vals=q2.detach().cpu().numpy(),
                         Q1ExtractedMemory=q1_extracted_memory.mean(dim=1).detach().cpu().numpy(),
                         Q2ExtractedMemory=q2_extracted_memory.mean(dim=1).detach().cpu().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        o, h_o, h_a, h_o_len = data['obs'], data['hist_obs'], data['hist_act'], data['hist_obs_len']
        a, a_extracted_memory = ac.pi(o, h_o, h_a, h_o_len)
        q1_pi, _ = ac.q1(o, a, h_o, h_a, h_o_len)
        loss_info = dict(ActExtractedMemory=a_extracted_memory.mean(dim=1).detach().cpu().numpy())
        return -q1_pi.mean(), loss_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    def update(data, timer):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **loss_info)

        # Possibly update pi and target networks
        if timer % policy_delay == 0:
            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi, loss_info_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Record things
            logger.store(LossPi=loss_pi.item(), **loss_info_pi)

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, o_buff, a_buff, o_buff_len, noise_scale, device=None):
        h_o = torch.tensor(o_buff).view(1, o_buff.shape[0], o_buff.shape[1]).float().to(device)
        h_a = torch.tensor(a_buff).view(1, a_buff.shape[0], a_buff.shape[1]).float().to(device)
        h_l = torch.tensor([o_buff_len]).float().to(device)
        with torch.no_grad():
            a = ac.act(torch.as_tensor(o, dtype=torch.float32).view(1, -1).to(device),
                       h_o, h_a, h_l).reshape(act_dim)
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0

            if max_hist_len > 0:
                o_buff = np.zeros([max_hist_len, obs_dim])
                a_buff = np.zeros([max_hist_len, act_dim])
                o_buff[0, :] = o
                o_buff_len = 0
            else:
                o_buff = np.zeros([1, obs_dim])
                a_buff = np.zeros([1, act_dim])
                o_buff_len = 0

            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                a = get_action(o, o_buff, a_buff, o_buff_len, 0, device)
                o2, r, d, _ = test_env.step(a)

                ep_ret += r
                ep_len += 1
                # Add short history
                if max_hist_len != 0:
                    if o_buff_len == max_hist_len:
                        o_buff[:max_hist_len - 1] = o_buff[1:]
                        a_buff[:max_hist_len - 1] = a_buff[1:]
                        o_buff[max_hist_len - 1] = list(o)
                        a_buff[max_hist_len - 1] = list(a)
                    else:
                        o_buff[o_buff_len + 1 - 1] = list(o)
                        a_buff[o_buff_len + 1 - 1] = list(a)
                        o_buff_len += 1
                o = o2

            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment

    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    past_t = 0
    o, ep_ret, ep_len = env.reset(), 0, 0

    if max_hist_len > 0:
        o_buff = np.zeros([max_hist_len, obs_dim])
        a_buff = np.zeros([max_hist_len, act_dim])
        o_buff[0, :] = o
        o_buff_len = 0
    else:
        o_buff = np.zeros([1, obs_dim])
        a_buff = np.zeros([1, act_dim])
        o_buff_len = 0

    if resume_exp_dir is not None:
        # Find the latest checkpoint
        resume_checkpoint_path = osp.join(resume_exp_dir, "pyt_save")
        checkpoint_files = os.listdir(resume_checkpoint_path)
        latest_context_version = np.max([int(f_name.split('-')[3]) for f_name in checkpoint_files if
                                         'context' in f_name and 'verified' in f_name])
        latest_model_version = np.max([int(f_name.split('-')[3]) for f_name in checkpoint_files if
                                       'model' in f_name and 'verified' in f_name])
        if latest_context_version != latest_model_version:
            latest_version = np.min([latest_context_version, latest_model_version])
        else:
            latest_version = latest_context_version
        latest_context_checkpoint_file_name = 'checkpoint-context-Step-{}-verified.pt'.format(latest_version)
        latest_model_checkpoint_file_name = 'checkpoint-model-Step-{}-verified.pt'.format(latest_version)
        latest_context_checkpoint_file_path = osp.join(resume_checkpoint_path, latest_context_checkpoint_file_name)
        latest_model_checkpoint_file_path = osp.join(resume_checkpoint_path, latest_model_checkpoint_file_name)

        # Load the latest checkpoint
        context_checkpoint = torch.load(latest_context_checkpoint_file_path)
        model_checkpoint = torch.load(latest_model_checkpoint_file_path)

        # Restore experiment context
        logger.epoch_dict = context_checkpoint['logger_epoch_dict']
        replay_buffer = context_checkpoint['replay_buffer']
        start_time = context_checkpoint['start_time']
        past_t = context_checkpoint['t'] + 1  # Crucial add 1 step to t to avoid repeating.

        # Restore model
        ac.load_state_dict(model_checkpoint['ac_state_dict'])
        ac_targ.load_state_dict(model_checkpoint['target_ac_state_dict'])
        pi_optimizer.load_state_dict(model_checkpoint['pi_optimizer_state_dict'])
        q_optimizer.load_state_dict(model_checkpoint['q_optimizer_state_dict'])

    print("past_t={}".format(past_t))
    # Main loop: collect experience in env and update/log each epoch
    for t in range(past_t, total_steps):  # Start from the step after resuming.
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = get_action(o, o_buff, a_buff, o_buff_len, act_noise, device)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)

        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Add short history
        if max_hist_len != 0:
            if o_buff_len == max_hist_len:
                o_buff[:max_hist_len - 1] = o_buff[1:]
                a_buff[:max_hist_len - 1] = a_buff[1:]
                o_buff[max_hist_len - 1] = list(o)
                a_buff[max_hist_len - 1] = list(a)
            else:
                o_buff[o_buff_len + 1 - 1] = list(o)
                a_buff[o_buff_len + 1 - 1] = list(a)
                o_buff_len += 1

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

            if max_hist_len > 0:
                o_buff = np.zeros([max_hist_len, obs_dim])
                a_buff = np.zeros([max_hist_len, act_dim])
                o_buff[0, :] = o
                o_buff_len = 0
            else:
                o_buff = np.zeros([1, obs_dim])
                a_buff = np.zeros([1, act_dim])
                o_buff_len = 0

            # Store checkpoint at the end of trajectory, so there is no need to store env as resume env in PyBullet is problematic.
            # Save the context of the learning and learned models
            fpath = 'pyt_save'
            fpath = osp.join(logger.output_dir, fpath)
            os.makedirs(fpath, exist_ok=True)
            old_checkpoints = os.listdir(fpath)  # Cache old checkpoints to delete later
            # Separately save context and model to reduce disk space usage.
            context_fname = 'checkpoint-context-' + (
                'Step-%d' % t if t is not None else '') + '.pt'
            model_fname = 'checkpoint-model-' + ('Step-%d' % t if t is not None else '') + '.pt'

            context_elements = {'env': env, 'replay_buffer': replay_buffer,
                                'logger_epoch_dict': logger.epoch_dict,
                                'start_time': start_time, 't': t}
            model_elements = {'ac_state_dict': ac.state_dict(),
                              'target_ac_state_dict': ac_targ.state_dict(),
                              'pi_optimizer_state_dict': pi_optimizer.state_dict(),
                              'q_optimizer_state_dict': q_optimizer.state_dict()}
            context_fname = osp.join(fpath, context_fname)
            torch.save(context_elements, context_fname)
            model_fname = osp.join(fpath, model_fname)
            torch.save(model_elements, model_fname)
            # Rename the file to verify the completion of the saving.
            verified_context_fname = osp.join(fpath, 'checkpoint-context-' + (
                'Step-%d' % t if t is not None else '') + '-verified.pt')
            verified_model_fname = osp.join(fpath, 'checkpoint-model-' + (
                'Step-%d' % t if t is not None else '') + '-verified.pt')
            os.rename(context_fname, verified_context_fname)
            os.rename(model_fname, verified_model_fname)
            # Remove old checkpoint
            for old_f in old_checkpoints:
                os.remove(osp.join(fpath, old_f))

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch_with_history(batch_size, max_hist_len)
                batch = {k: v.to(device) for k, v in batch.items()}
                update(data=batch, timer=j)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('Q1ExtractedMemory', with_min_and_max=True)
            logger.log_tabular('Q2ExtractedMemory', with_min_and_max=True)
            logger.log_tabular('ActExtractedMemory', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)

            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


def str2bool(v):
    """Function used in argument parser for converting string to bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def list2tuple(v):
    return tuple(v)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_exp_dir', type=str, default=None, help="The directory of the resuming experiment.")
    parser.add_argument('--env_name', type=str, default='HalfCheetahBulletEnv-v0')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--max_hist_len', type=int, default=5)
    parser.add_argument('--partially_observable', type=str2bool, nargs='?', const=True, default=False, help="Using POMDP")
    parser.add_argument('--pomdp_type',
                        choices=['remove_velocity', 'flickering', 'random_noise', 'random_sensor_missing',
                                 'remove_velocity_and_flickering', 'remove_velocity_and_random_noise',
                                 'remove_velocity_and_random_sensor_missing', 'flickering_and_random_noise',
                                 'random_noise_and_random_sensor_missing', 'random_sensor_missing_and_random_noise'],
                        default='remove_velocity')
    parser.add_argument('--flicker_prob', type=float, default=0.2)
    parser.add_argument('--random_noise_sigma', type=float, default=0.1)
    parser.add_argument('--random_sensor_missing_prob', type=float, default=0.1)
    parser.add_argument('--use_double_critic', type=str2bool, nargs='?', const=True, default=True,
                        help="Using double critic")
    parser.add_argument('--use_target_policy_smooth', type=str2bool, nargs='?', const=True, default=True,
                        help="Using target policy smoothing")
    parser.add_argument('--critic_mem_pre_lstm_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--critic_mem_lstm_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--critic_mem_after_lstm_hid_size', type=int, nargs="+", default=[])
    parser.add_argument('--critic_cur_feature_hid_sizes', type=int, nargs="?", default=[128, 128])
    parser.add_argument('--critic_post_comb_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--critic_hist_with_past_act', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--actor_mem_pre_lstm_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--actor_mem_lstm_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--actor_mem_after_lstm_hid_size', type=int, nargs="+", default=[])
    parser.add_argument('--actor_cur_feature_hid_sizes', type=int, nargs="?", default=[128, 128])
    parser.add_argument('--actor_post_comb_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--actor_hist_with_past_act', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--exp_name', type=str, default='lstm_td3')
    parser.add_argument("--data_dir", type=str, default='spinup_data_lstm_gate')
    args = parser.parse_args()

    # Interpret without current feature extraction.
    if args.critic_cur_feature_hid_sizes is None:
        args.critic_cur_feature_hid_sizes = []
    if args.actor_cur_feature_hid_sizes is None:
        args.actor_cur_feature_hid_sizes = []


    # Set log data saving directory
    if args.resume_exp_dir is None:
        # data_dir = osp.join(
        #     osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))))),
        #     args.data_dir)
        # data_dir = osp.join(
        #         osp.dirname('/scratch/lingheng/'),
        #         args.data_dir)
        data_dir = osp.join(
            osp.dirname('F:/scratch/lingheng/'),
            args.data_dir)
        logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir, datestamp=True)
    else:
        # Load config_json
        resume_exp_dir = args.resume_exp_dir
        config_path = osp.join(args.resume_exp_dir, 'config.json')
        with open(osp.join(args.resume_exp_dir, "config.json"), 'r') as config_file:
            config_json = json.load(config_file)
        # Update resume_exp_dir value as default is None.
        config_json['resume_exp_dir'] = resume_exp_dir
        # Print config_json
        output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
        print(colorize('Loading config:\n', color='cyan', bold=True))
        print(output)
        # Restore the hyper-parameters
        logger_kwargs = config_json["logger_kwargs"]   # Restore logger_kwargs
        config_json.pop('logger', None)                # Remove logger from config_json
        args = json.loads(json.dumps(config_json), object_hook=lambda d: namedtuple('args', d.keys())(*d.values()))


    lstm_td3(resume_exp_dir=args.resume_exp_dir,
             env_name=args.env_name,
             gamma=args.gamma, seed=args.seed, epochs=args.epochs,
             max_hist_len=args.max_hist_len,
             partially_observable=args.partially_observable,
             pomdp_type=args.pomdp_type,
             flicker_prob=args.flicker_prob,
             random_noise_sigma=args.random_noise_sigma,
             random_sensor_missing_prob=args.random_sensor_missing_prob,
             use_double_critic=args.use_double_critic,
             use_target_policy_smooth=args.use_target_policy_smooth,
             critic_mem_pre_lstm_hid_sizes=tuple(args.critic_mem_pre_lstm_hid_sizes),
             critic_mem_lstm_hid_sizes=tuple(args.critic_mem_lstm_hid_sizes),
             critic_mem_after_lstm_hid_size=tuple(args.critic_mem_after_lstm_hid_size),
             critic_cur_feature_hid_sizes=tuple(args.critic_cur_feature_hid_sizes),
             critic_post_comb_hid_sizes=tuple(args.critic_post_comb_hid_sizes),
             actor_mem_pre_lstm_hid_sizes=tuple(args.actor_mem_pre_lstm_hid_sizes),
             actor_mem_lstm_hid_sizes=tuple(args.actor_mem_lstm_hid_sizes),
             actor_mem_after_lstm_hid_size=tuple(args.actor_mem_after_lstm_hid_size),
             actor_cur_feature_hid_sizes=tuple(args.actor_cur_feature_hid_sizes),
             actor_post_comb_hid_sizes=tuple(args.actor_post_comb_hid_sizes),
             actor_hist_with_past_act=args.actor_hist_with_past_act,
             logger_kwargs=logger_kwargs)
