from irl.utils import *

def estimate_advantages(rewards, masks, values, gamma, tau, device):
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns


def calculate_return(rewards, pred_len, bs, gamma):

    G = torch.zeros_like(rewards)
    for i in reversed(range(pred_len)):
        if i == pred_len-1:
            G[i * bs:(i + 1) * bs] = rewards[i * bs:(i + 1) * bs]
        else:
            G[i * bs:(i + 1) * bs] = rewards[i * bs:(i + 1) * bs] + gamma * G[(i + 1) * bs:(i + 2) * bs]
    return G    #(bx12,)
