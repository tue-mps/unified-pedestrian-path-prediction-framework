import argparse
import os
import torch
import numpy as np
import random
import torch.nn.functional as F

from attrdict import AttrDict

from irl.data.loader import data_loader
from irl.models import Policy
from irl.losses import displacement_error, final_displacement_error
from irl.utils import relative_to_abs, get_dset_path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='eth', type=str)
parser.add_argument('--model_path', default="../models/irl-models", type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--model_average', default=True, type=bool)
parser.add_argument('--runs', type=int, default=5, help='number of models to compute average')
parser.add_argument('--prediction_steps', default=None, type=int)
parser.add_argument('--noise', default=False, type=bool)             # add noise to deterministic models to add stochasticity

# additional params
seeding = 0

if seeding is not None:
    torch.manual_seed(seeding)
    np.random.seed(seeding)
    random.seed(seeding)


def get_policy(checkpoint):
    args = AttrDict(checkpoint['args'])
    policy_net = Policy(16, 2, log_std=args.log_std)
    policy_net.load_state_dict(checkpoint['policy_net_state'])
    policy_net.cuda()
    policy_net.eval()
    return policy_net


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def compute_av_std(ADE, FDE, minADE, minFDE):
    ADE = torch.stack(ADE)
    FDE = torch.stack(FDE)
    minADE = torch.stack(minADE)
    minFDE = torch.stack(minFDE)

    av_ade = torch.mean(ADE)
    std_ade = torch.std(ADE, unbiased=False)
    low_ade = torch.min(ADE)

    av_fde = torch.mean(FDE)
    std_fde = torch.std(FDE, unbiased=False)
    low_fde = torch.min(FDE)

    av_min_ade = torch.mean(minADE)
    std_min_ade = torch.std(minADE, unbiased=False)
    low_min_ade = torch.min(minADE)

    av_min_fde = torch.mean(minFDE)
    std_min_fde = torch.std(minFDE, unbiased=False)
    low_min_fde = torch.min(minFDE)

    return av_ade, av_fde, av_min_ade, av_min_fde, std_ade, std_fde, std_min_ade, std_min_fde, low_ade, low_fde, low_min_ade, low_min_fde


def add_noise_state(state, device):
    # add noise from gaussian to state (b,16)
    noise_dim = 2
    noise_std = 0.05
    pad_dim = state.shape[1] - noise_dim
    noise_shape = (state.shape[0], noise_dim)
    noise = torch.randn(noise_shape).to(device)
    noise = noise * noise_std

    pad = (pad_dim, 0, 0, 0)
    noise = F.pad(noise, pad, "constant", 0)  # effectively zero padding

    state = state + noise
    return state

def create_fake_trajectories(obs_traj_rel, policy, pred_len, device, randomness_definition, mean_action, noise):
    obs_len = obs_traj_rel.shape[0]

    state = obs_traj_rel.permute(1, 0, 2)
    state = torch.flatten(state, 1, 2)
    fake_traj = state                      # (b, 16) in x,y,x,y format

    for step in range(pred_len):
        if mean_action is False and randomness_definition == 'stochastic':
            action = policy.select_action(state)
        else:
            if noise:
                state = add_noise_state(state, device)                                          # this is to add noise remove if no worky or make better if worky
            action, _, _ = policy(state)    #meanaction for deterministic
        fake_traj = torch.cat((fake_traj, action), dim=1)
        next_state = torch.cat((state, action), dim=1)[:, -obs_len * 2:]
        state = next_state

    fake_traj = fake_traj[:,-pred_len*2:]   # (b, 24)

    fake_traj = torch.reshape(fake_traj, (fake_traj.shape[0], pred_len, 2))  # (b, 12, 2)
    pred_traj_fake_rel = fake_traj.permute(1, 0, 2)                          # (12, b, 2)

    return pred_traj_fake_rel   # (12, b, 2)


def evaluate_irl(args, loader, policy_net, num_samples, mean_action, noise, device):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):

                pred_traj_fake_rel = create_fake_trajectories(obs_traj_rel, policy_net, args.pred_len, device, args.randomness_definition, mean_action, noise)

                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde


def main(args):
    if args.model_average:
        args.model_path = "../models/model_average"
        ADE = []
        FDE = []
        minADE = []
        minFDE = []
    else:
        args.model_path = "../models/irl-models"
        list_to_sort = []

    if seeding:
        torch.manual_seed(seeding)
        np.random.seed(seeding)

    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    print("Evaluating IRL model \n")
    path_counter = 0
    for path in paths:
        path_counter = path_counter + 1
        checkpoint = torch.load(path)
        policy_net = get_policy(checkpoint)
        _args = AttrDict(checkpoint['args'])
        if args.prediction_steps is not None:
            _args.pred_len = args.prediction_steps
        if args.model_average is False:
            print("\n path: ", path)
        temp_path = path
        model_name = path.split("saved_model_")[1].split(".p")[0].ljust(30)
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        ade, fde = evaluate_irl(_args, loader, policy_net, 1, mean_action=True, noise=False, device=device)
        if args.model_average is False:
            print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
                _args.dataset_name, _args.pred_len, ade, fde))
        if args.model_average:
            ADE.append(ade)
            FDE.append(fde)

        ## best k predicting
        min_ade, min_fde = evaluate_irl(_args, loader, policy_net, args.num_samples, mean_action=False, noise=args.noise, device=device)
        if args.model_average is False:
            print('Dataset: {}, Pred Len: {}, k: {}, minADE: {:.2f}, minFDE: {:.2f}'.format(
                _args.dataset_name, _args.pred_len, args.num_samples, min_ade, min_fde))
        if args.model_average:
            minADE.append(min_ade)
            minFDE.append(min_fde)
        else:
            ade_2d = "{:.2f}".format(ade.item())
            fde_2d = "{:.2f}".format(fde.item())
            minade_2d = "{:.2f}".format(min_ade.item())
            minfde_2d = "{:.2f}".format(min_fde.item())
            list_entry = (model_name, ade_2d, fde_2d, minade_2d, minfde_2d)
            list_to_sort.append(list_entry)

        if args.model_average is True and path_counter == args.runs:
            print("\nAverage for path: ", temp_path)
            av_ade, av_fde, av_min_ade, av_min_fde, std_ade, std_fde, std_min_ade, std_min_fde, low_ade, low_fde, low_min_ade, low_min_fde = compute_av_std(ADE, FDE, minADE, minFDE)

            print('Dataset: {}, Pred Len: {}, mean ADE: {:.2f}, std ADE: {:.4f}, low ADE: {:.2f}, mean FDE: {:.2f}, std FDE: {:.4f}, low FDE: {:.2f}'.format(
                _args.dataset_name, _args.pred_len, av_ade, std_ade, low_ade, av_fde, std_fde, low_fde))
            print('Dataset: {}, Pred Len: {}, k: {}, mean minADE: {:.2f}, std minADE: {:.4f}, low minADE: {:.2f}, mean minFDE: {:.2f}, std minFDE: {:.4f}, low minFDE: {:.2f}'.format(
                _args.dataset_name, _args.pred_len, args.num_samples, av_min_ade, std_min_ade, low_min_ade, av_min_fde, std_min_fde, low_min_fde))

            path_counter = 0
            ADE = []
            FDE = []
            minADE = []
            minFDE = []
            if seeding:
                torch.manual_seed(seeding)
                np.random.seed(seeding)


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("device: ", device)
    main(args)
