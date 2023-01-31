
from irl.replay_memory import Memory
from irl.utils import *
import os
import torch.nn.functional as F
os.environ["OMP_NUM_THREADS"] = "1"

def add_noise_to_state(state):
    noise_dim = 2
    noise_std = 0.05
    pad_dim = state.shape[1] - noise_dim
    noise_shape = (state.shape[0], noise_dim)
    device = state.device
    noise = torch.randn(noise_shape).to(device)
    noise = noise * noise_std

    pad = (pad_dim, 0, 0, 0)
    noise = F.pad(noise, pad, "constant", 0)  # effectively zero padding

    noisy_state = state + noise
    return noisy_state


def collect_samples(args, env, policy, custom_reward, device, mean_action):

    total_paths = env.total_paths
    memory = Memory()

    with torch.no_grad():
        state = env.reset()             # take new initial state (observation) of shape (b, 16)
        state_0 = state.clone()
        bs = state.shape[0]             # batch size
        rewards = []
        states = []
        actions = []
        reward_full = torch.zeros(bs)   #(b,) of zeros
        ground_truth = env.collect_expert()   # (b,40) if single step, (bx12,18) if multistep
        ts = 0          # timestep for selecting correct ground truth

        done = False
        while not done:

            if mean_action:
                if args.trainable_noise == True:
                    state = add_noise_to_state(state)
                action, _, _ = policy(state)                  # action is of shape (b, 2)
            else:
                action = policy.select_action(state)

            # save action
            actions.append(action)
            states.append(state)

            next_state, reward, done, = env.step(state, action)

            if custom_reward is not None:
                if args.step_definition == 'multi':
                    gt = ground_truth[ts * bs:(ts + 1) * bs, :] # take the ground truth of the correct timestep
                    reward = torch.squeeze(custom_reward(args, state, action, gt), dim=1)
                    rewards.append(reward)

            if done:
                action_full = torch.cat(actions, dim=1)

                if custom_reward is not None:
                    if args.step_definition == 'single':
                        gt = ground_truth
                        reward_full = torch.squeeze(custom_reward(args, state_0, action_full, gt), dim=1)

                if args.step_definition == 'multi':
                    rewards = torch.cat(rewards, dim=0)  # (bx12,)
                states = torch.cat(states, dim=0)        # (bx12, 16)
                actions = torch.cat(actions, dim=0)      # (bx12, 2)
                memory.push(state_0, action_full, reward_full, states, actions, rewards)   # initial state, 12dim action, reward (single), all intermediate states, all intermediate actions, rewards (multi)
                break

            state = next_state
            ts = ts + 1

    return memory

def reshape_batch(batch):
    states = torch.stack(batch.state)
    actions = torch.stack(batch.action, dim=0)
    rewards = torch.stack(batch.reward, dim=0)

    states = torch.flatten(states.permute(1,0,2), 0, 1)
    actions = torch.flatten(actions.permute(1,0,2), 0, 1)
    rewards = torch.flatten(rewards.permute(1, 0))

    states_all = batch.states_all
    actions_all = batch.actions_all
    rewards_all = batch.rewards_all

    batch = (states, actions, rewards, states_all, actions_all, rewards_all)
    return batch


class Agent:

    def __init__(self, args, env, policy, device, custom_reward=None):
        self.args = args
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward

    def collect_samples(self, mean_action=False):

        memory = collect_samples(self.args, self.env, self.policy, self.custom_reward, self.device, mean_action)
        batch = memory.sample()
        batch = reshape_batch(batch)

        return batch

