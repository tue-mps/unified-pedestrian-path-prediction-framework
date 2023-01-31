import argparse
from irl.utils import *
from irl.models import Policy, Discriminator, Value
from torch import nn
from irl.data.loader import data_loader
from irl.update_parameters import discriminator_step, reinforce_step, generator_step
from irl.accuracy import check_accuracy
from irl.agent import Agent
from irl.environment import Environment
from torch.utils.tensorboard import SummaryWriter

from scripts.evaluate_model import evaluate_irl

"""arguments"""
parser = argparse.ArgumentParser(description='PyTorch Unified PPP framework')

parser.add_argument('--randomness_definition', default='stochastic',  type=str, help='either stochastic or deterministic')
parser.add_argument('--step_definition', default='multi',  type=str, help='either single or multi')
parser.add_argument('--loss_definition', default='discriminator',  type=str, help='either discriminator or l2')
parser.add_argument('--discount_factor', type=float, default=0.0, help='discount factor gamma, value between 0.0 and 1.0')

parser.add_argument('--training_algorithm', default='reinforce',  type=str, help='choose which RL updating algorithm, either "reinforce", "baseline" or "ppo" or "ppo_only"')
parser.add_argument('--trainable_noise', type=bool, default=False, help='add a noise to the input during training')
parser.add_argument('--ppo-iterations', type=int, default=1, help='number of ppo iterations (default=1)')
parser.add_argument('--ppo-clip', type=float, default=0.2, help='amount of ppo clipping (default=0.2)')
parser.add_argument('--learning-rate', type=float, default=1e-5, metavar='G', help='learning rate (default: 1e-5)')
parser.add_argument('--batch_size', default=8, type=int, help='number of sequences in a batch (can be multiple paths)')
parser.add_argument('--log-std', type=float, default=-2.99, metavar='G', help='log std for the policy (default=-0.0)')
parser.add_argument('--num_epochs', default=200, type=int, help='number of times the model sees all data')

parser.add_argument('--seeding', type=bool, default=True, help='turn seeding on or off')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--multiple_executions', type=bool, default=True, help='turn multiple runs on or off')
parser.add_argument('--runs', type=int, default=5, help='number of times the script runs')
parser.add_argument('--all_datasets', type=bool, default=True, help='run the script for all 5 datasets at once or not')
parser.add_argument('--dataset_name', default='eth',  type=str, help='choose which dataset to train for')
parser.add_argument('--check_testset', type=bool, default=True, help='also evaluate on the testset, next to validation set')
parser.add_argument('--output_dir', default=os.getcwd(), help='path where models are saved (default=current directory)')
parser.add_argument('--save_model_name_ADE', default="saved_model_ADE", help='name of the saved model')
parser.add_argument('--save_model_name_FDE', default="saved_model_FDE", help='name of the saved model')
parser.add_argument('--num_samples_check', default=5000, type=int, help='limit the nr of samples during metric calculation')
parser.add_argument('--check_validation_every', default=1, type=int, help='check the metrics on the validation dataset every X epochs')

parser.add_argument('--obs_len', default=8, type=int, help='how many timesteps used for observation (default=8)')
parser.add_argument('--pred_len', default=12, type=int, help='how many timesteps used for prediction (default=12)')
parser.add_argument('--discriminator_steps', default=1, type=int, help='how many discriminator updates per iteration')
parser.add_argument('--policy_steps', default=1, type=int, help='how many policy updates per iteration')
parser.add_argument('--loader_num_workers', default=0, type=int, help='number cpu/gpu processes (default=0)')
parser.add_argument('--skip', default=1, type=int, help='used for skipping sequences (default=1)')
parser.add_argument('--delim', default='\t', help='how to read the data text file spacing')
parser.add_argument('--l2_loss_weight', default=0, type=float, help='l2 loss multiplier (default=0)')
parser.add_argument('--use_gpu', default=1, type=int)                   # use gpu, if 0, use cpu only
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--load_saved_model', default=None, metavar='G', help='path of pre-trained model')

args = parser.parse_args()


def main_loop():

    """"definitions"""
    if args.randomness_definition == 'stochastic':
        mean_action = False
    elif args.randomness_definition == 'deterministic':
        mean_action = True
    else:
        print("Wrong definition for randomness, please choose either stochastic or deterministic")


    """seeding"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True           # what does this do again?
    torch.backends.cudnn.benchmark = False

    """dtype and device"""
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    print("device: ", device)

    """models"""
    policy_net = Policy(16, 2, log_std=args.log_std)     # 16, 2
    disc_single = Discriminator(40)
    disc_multi = Discriminator(18)

    if args.step_definition == 'multi':
        discriminator_net = disc_multi
    elif args.step_definition == 'single':
        discriminator_net = disc_single      #changed from single as experiment
    policy_net.to(device)
    policy_net.type(dtype).train()
    discriminator_net.to(device)
    discriminator_net.type(dtype).train()
    print("Policy_net: ", policy_net)
    print("Discriminator_net: ", discriminator_net)

    if (args.training_algorithm == 'baseline' or args.training_algorithm == 'ppo' or args.training_algorithm == 'ppo_only'):
        value_net = Value(16)
        value_net.to(device)
        value_net.type(dtype).train()
        print("Value_net: ", value_net)
    else:
        value_net = None

    """optimizers"""
    policy_opt = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    disc_lr = args.learning_rate
    discriminator_opt = torch.optim.Adam(discriminator_net.parameters(), lr=disc_lr)
    discriminator_crt = nn.BCELoss()
    custom_reward = nn.BCELoss(reduction='none')
    if (args.training_algorithm == 'baseline' or args.training_algorithm == 'ppo' or args.training_algorithm == 'ppo_only'):
        value_opt = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
        value_crt = nn.MSELoss()
    else:
        value_opt = None
        value_crt = None

    """datasets"""
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')
    test_path = get_dset_path(args.dataset_name, 'test')
    print("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    train_loader_len = len(train_loader)
    print("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)
    val_loader_len = len(val_loader)
    print("Initializing test dataset")
    _, test_loader = data_loader(args, test_path)
    test_loader_len = len(test_loader)

    """loading the model"""
    if args.load_saved_model is None:
        """we want to store a model optimized for ADE metric"""
        saved_model_ADE = {
            'args': args.__dict__,
            'epoch': 0,
            'ADE_train': [],
            'FDE_train': [],
            'ADE_val': [],
            'FDE_val': [],
            'policy-loss_train': [],
            'policy-loss_val': [],
            'policy_net_state': None,
            'policy_opt_state': None,
            'discriminator-loss_train': [],
            'discriminator-loss_val': [],
            'discriminator_net_state': None,
            'discriminator_opt_state': None,
            'value-loss_train': [],
            'value-loss_val': [],
        }
        epoch = saved_model_ADE['epoch']
        save_model_path_ADE = os.path.join(args.output_dir, '%s.pt' % args.save_model_name_ADE)

        """We also want a separate model that is optimized for FDE metric"""
        saved_model_FDE = {
            'args': args.__dict__,
            'epoch': 0,
            'ADE_train': [],
            'FDE_train': [],
            'ADE_val': [],
            'FDE_val': [],
            'policy-loss_train': [],
            'policy-loss_val': [],
            'policy_net_state': None,
            'policy_opt_state': None,
            'discriminator-loss_train': [],
            'discriminator-loss_val': [],
            'discriminator_net_state': None,
            'discriminator_opt_state': None,
            'value-loss_train': [],
            'value-loss_val': [],
        }
        save_model_path_FDE = os.path.join(args.output_dir, '%s.pt' % args.save_model_name_FDE)

    else:
        saved_model = torch.load(args.load_saved_model)
        policy_net.load_state_dict(saved_model['policy_net_state'])
        policy_opt.load_state_dict(saved_model['policy_opt_state'])
        discriminator_net.load_state_dict(saved_model['discriminator_net_state'])
        discriminator_opt.load_state_dict(saved_model['discriminator_opt_state'])


    """custom reward for policy"""
    def expert_reward(args, state, action, gt=0):     # probs separate function for discount

        state_action = torch.cat((state, action), dim=1)  # (b, 16) + (b, 24) = (b, 40)
        if args.loss_definition == 'discriminator':
            disc_out = discriminator_net(state_action)
            labels = torch.ones_like(disc_out)
            expert_reward = -custom_reward(disc_out, labels)  # pytorch nn.BCELoss() already has a -

        elif args.loss_definition == 'l2':
            l2 = (gt - state_action)**2  # (b, 40) - (b, 40) ** 2 (for single, (b,18) for multi)
            l2 = l2[:, 16:]              # test to only include action difference instead of state-action
            expert_reward = -l2.sum(dim=1, keepdim=True)    #div dim action space
        else:
            print("Wrong definition for loss, please choose either discriminator or l2")

        return expert_reward    # tensor(b,1)


    """create agent"""
    env = Environment(args, device)
    agent = Agent(args, env, policy_net, device, custom_reward=expert_reward)


    """update parameters function"""
    def update_params(args, batch, expert, train):

        loss_policy = 0
        loss_discriminator = 0
        loss_value = 0

        states, actions, rewards, states_all, actions_all, rewards_all = batch

        if args.loss_definition == 'discriminator':
            """perform discriminator update"""
            for _ in range(args.discriminator_steps):
                if args.step_definition == 'single':
                    expert_state_actions = expert
                    pred_state_actions = torch.cat([states, actions], dim=1)
                elif args.step_definition == 'multi':
                    expert_state_actions = expert   # (bx12, 18)
                    pred_state_actions = torch.cat([states_all[0], actions_all[0]], dim=1)  #(bx12, 18)
                discriminator_loss = discriminator_step(discriminator_net, discriminator_opt, discriminator_crt, expert_state_actions, pred_state_actions, device, train)
                loss_discriminator += discriminator_loss

        """perform policy (REINFORCE) update"""
        for _ in range(args.policy_steps):
            policy_loss, value_loss = reinforce_step(args, policy_net, policy_opt, expert_reward, states_all, actions_all,
                                         rewards_all, rewards, expert, train, value_net, value_opt, value_crt)

            loss_policy += policy_loss
            loss_value += value_loss

        return loss_policy/args.policy_steps, loss_discriminator/args.discriminator_steps, loss_value/args.policy_steps


    def train_loop():
        t0 = time.time()
        for epoch in range(args.num_epochs):     # epochs
            t1 = time.time()
            print("\nEpoch: ", epoch, "/", args.num_epochs)

            """perform training steps"""
            train = True
            loss_policy = torch.zeros(1, device=device)
            loss_discriminator = torch.zeros(1, device=device)
            loss_value = torch.zeros(1, device=device)

            for batch_input in train_loader:
                with torch.autograd.set_detect_anomaly(True):
                    env.generate(batch_input)                                   # sets a batch of observed trajectories
                    batch = agent.collect_samples(mean_action=mean_action)      # batch contains a tensor of states (8 steps), a tensor of actions (12 steps) and a tensor of rewards (1 for the whole trajectory)
                    expert = env.collect_expert()                               # the expert is a batch of full ground truth trajectories

                    policy_loss, discriminator_loss, value_loss = update_params(args, batch, expert, train)
                    loss_policy += policy_loss
                    loss_discriminator += discriminator_loss
                    loss_value += value_loss

            metrics_train = check_accuracy(args, train_loader, policy_net, args.pred_len, device, limit=False)       # limit=true causes sinusoidal train ADE

            loss_policy = loss_policy / train_loader_len
            loss_discriminator = loss_discriminator / train_loader_len
            loss_value = loss_value / train_loader_len

            writer.add_scalar('train_loss_policy', loss_policy.item(), epoch)
            writer.add_scalar('train_loss_discriminator', loss_discriminator.item(), epoch)
            writer.add_scalar('train_loss_value', loss_value.item(), epoch)
            writer.add_scalar('ADE_train', metrics_train['ade'], epoch)
            writer.add_scalar('FDE_train', metrics_train['fde'], epoch)

            print('train loss_policy: ', loss_policy.item())
            print('train loss_discriminator: ', loss_discriminator.item())
            print('train loss_value: ', loss_value.item())
            print('train ADE: ', metrics_train['ade'])
            print('train FDE: ', metrics_train['fde'])

            if epoch % args.check_validation_every == 0:

                """perform validation steps"""
                train = False
                loss_policy_val = torch.zeros(1, device=device)
                loss_discriminator_val = torch.zeros(1, device=device)
                loss_value_val = torch.zeros(1, device=device)
                for batch_input in val_loader:
                    env.generate(batch_input)
                    batch = agent.collect_samples(mean_action=mean_action)
                    expert = env.collect_expert()

                    policy_loss_val, discriminator_loss_val, value_loss_val = update_params(args, batch, expert, train)
                    loss_policy_val += policy_loss_val
                    loss_discriminator_val += discriminator_loss_val
                    loss_value_val += value_loss_val

                metrics_validation = check_accuracy(args, val_loader, policy_net, args.pred_len, device, limit=False)

                ### test set check
                if args.check_testset is True:
                    test_ade, test_fde = evaluate_irl(args, test_loader, policy_net, num_samples=1, mean_action=True, noise=args.trainable_noise, device=device)
                    test_minade, test_minfde = evaluate_irl(args, test_loader, policy_net, num_samples=20, mean_action=False, noise=args.trainable_noise, device=device)
                    writer.add_scalar('ADE_test', test_ade, epoch)
                    writer.add_scalar('FDE_test', test_fde, epoch)
                    writer.add_scalar('minADE_test', test_minade, epoch)
                    writer.add_scalar('minFDE_test', test_minfde, epoch)

                loss_policy_val = loss_policy_val / val_loader_len
                loss_discriminator_val = loss_discriminator_val / val_loader_len
                loss_value_val = loss_value_val / val_loader_len

                writer.add_scalar('validation_loss_policy', loss_policy_val.item(), epoch)
                writer.add_scalar('validation_loss_discriminator', loss_discriminator_val.item(), epoch)
                writer.add_scalar('validation_loss_value', loss_value_val.item(), epoch)
                writer.add_scalar('ADE_val', metrics_validation['ade'], epoch)
                writer.add_scalar('FDE_val', metrics_validation['fde'], epoch)


                print('validation loss_policy: ', loss_policy_val.item())
                print('validation loss_discriminator: ', loss_discriminator_val.item())
                print('validation loss_value: ', loss_value_val.item())
                print('validation ADE: ', metrics_validation['ade'])
                print('validation FDE: ', metrics_validation['fde'])

                if saved_model_ADE['ADE_val']:
                    min_ade = saved_model_ADE['ADE_val']  # both linear and non-linear
                else:
                    min_ade = metrics_validation['ade']
                if saved_model_FDE['FDE_val']:
                    min_fde = saved_model_FDE['FDE_val']  # both linear and non-linear
                else:
                    min_fde = metrics_validation['fde']

                if metrics_validation['ade'] <= min_ade:
                    print('New low for min ADE_val, model saved')
                    saved_model_ADE['epoch'] = epoch
                    saved_model_ADE['policy_net_state'] = policy_net.state_dict()
                    saved_model_ADE['policy_opt_state'] = policy_opt.state_dict()
                    saved_model_ADE['discriminator_net_state'] = discriminator_net.state_dict()
                    saved_model_ADE['discriminator_opt_state'] = discriminator_opt.state_dict()
                    saved_model_ADE['ADE_val'] = metrics_validation['ade']
                    saved_model_ADE['ADE_train'] = metrics_train['ade']
                    saved_model_ADE['FDE_val'] = metrics_validation['fde']
                    saved_model_ADE['FDE_train'] = metrics_train['fde']
                    saved_model_ADE['policy-loss_val'] = loss_policy_val.item()
                    saved_model_ADE['policy-loss_train'] = loss_policy.item()
                    saved_model_ADE['discriminator-loss_val'] = loss_discriminator_val.item()
                    saved_model_ADE['discriminator-loss_train'] = loss_discriminator.item()
                    saved_model_ADE['value-loss_val'] = loss_value_val.item()
                    saved_model_ADE['value-loss_train'] = loss_value.item()
                    torch.save(saved_model_ADE, save_model_path_ADE)
                if metrics_validation['fde'] <= min_fde:
                    print('New low for min FDE_val, model saved')
                    saved_model_FDE['epoch'] = epoch
                    saved_model_FDE['policy_net_state'] = policy_net.state_dict()
                    saved_model_FDE['policy_opt_state'] = policy_opt.state_dict()
                    saved_model_FDE['discriminator_net_state'] = discriminator_net.state_dict()
                    saved_model_FDE['discriminator_opt_state'] = discriminator_opt.state_dict()
                    saved_model_FDE['ADE_val'] = metrics_validation['ade']
                    saved_model_FDE['ADE_train'] = metrics_train['ade']
                    saved_model_FDE['FDE_val'] = metrics_validation['fde']
                    saved_model_FDE['FDE_train'] = metrics_train['fde']
                    saved_model_FDE['policy-loss_val'] = loss_policy_val.item()
                    saved_model_FDE['policy-loss_train'] = loss_policy.item()
                    saved_model_FDE['discriminator-loss_val'] = loss_discriminator_val.item()
                    saved_model_FDE['discriminator-loss_train'] = loss_discriminator.item()
                    saved_model_FDE['value-loss_val'] = loss_value_val.item()
                    saved_model_FDE['value-loss_train'] = loss_value.item()
                    torch.save(saved_model_FDE, save_model_path_FDE)

            t2 = time.time()
            print_time(t0, t1, t2, epoch)

    """execute train loop"""
    train_loop()


if args.all_datasets:
    datasets = ['eth', 'hotel', 'zara1', 'zara2', 'univ']
else:
    datasets = [args.dataset_name]

model_name_ADE_base = args.save_model_name_ADE
model_name_FDE_base = args.save_model_name_FDE

for set in datasets:
    args.dataset_name = set
    if args.multiple_executions:
        for i in range(args.runs):
            if args.seeding:
                args.seed = i
            model_name_ADE = model_name_ADE_base + '_' + set + '_run_' + str(i)
            model_name_FDE = model_name_FDE_base + '_' + set + '_run_' + str(i)
            tensorboard_name = '../tensorboard/' + set + '/run_' + str(i)
            args.save_model_name_ADE = model_name_ADE
            args.save_model_name_FDE = model_name_FDE
            writer = SummaryWriter(log_dir=tensorboard_name)
            print("Dataset: " + set + ". Script execution number: " + str(i))
            main_loop()
    else:
        model_name_ADE = model_name_ADE_base + '_' + set
        model_name_FDE = model_name_FDE_base + '_' + set
        tensorboard_name = '../tensorboard/' + set
        args.save_model_name_ADE = model_name_ADE
        args.save_model_name_FDE = model_name_FDE
        writer = SummaryWriter(log_dir=tensorboard_name)
        print("Dataset: " + set)
        main_loop()

