# Unified Pedestrian Path Prediction Framework

This is the code for the paper Unified Pedestrian Path Prediction Framework: A Comparison Study


## Setup
All code was developed and tested on Ubuntu 20.04 with Python 3.7 and PyTorch 1.10.

You can setup a virtual environment to run the code like this:

```bash
python3.7 -m venv env               # Create a virtual environment
source env/bin/activate           # Activate virtual environment
pip install -r requirements.txt   # Install dependencies
```


## Training new models
You can train your own model by following these instructions:

### Step 1: Data

The directory `datasets/<dataset_name>` contains train/ val/ and test/ splits. All the datasets are pre-processed to be in world coordinates i.e. in meters. We support five datasets ETH, ZARA1, ZARA2, HOTEL and UNIV. We use leave-one-out approach, train on 4 sets and test on the remaining set. We observe the trajectory for 8 times steps (3.2 seconds) and show prediction results for 8 (3.2 seconds) and 12 (4.8 seconds) time steps.

### Step 2: Train a model

Now you can train a new model by running the script:

```bash
python scripts/train_gail.py
```

By default this will train a model on ETH, periodically saving checkpoint files `saved_model_ADE.pt` and `saved_model_FDE.pt` for the lowest validation ADE/FDE separately, to the current working directory. The training script has a number of command-line flags that you can use to configure the model architecture, hyperparameters, and input / output settings:

### Framework settings

- `--randomness_definition`: Policy stochasticity: Stochastic or Deterministic
- `--step_definition`: Decision-making process: One-time or Sequential
- `--loss_definition`: Distance function: Squared L2-norm or Discriminator
- `--discount_factor`: Future awareness: γ = 0 or γ /= 0

### Hyper-parameters and algorithm settings

- `--training_algorithm`: Choose which RL updating algorithm, either "reinforce", "baseline" or "ppo" or "ppo_only". Default is 'reinforce'.
- `--trainable_noise`: Add a noise to the input during training. Default is False.
- `--ppo-iterations`: Number of PPO iterations. Default is 1.
- `--ppo-clip`: Amount of PPO clipping. Default is 0.2.
- `--learning-rate`: Learning rate. Default is 1e-5.
- `--batch_size`: Number of sequences in a batch (can be multiple paths). Default is 8.
- `--log-std`: Log std for the policy. Default is -2.99 (std = 0.05).
- `--num_epochs`: Number of times the model sees all data. Default is 200.

### Dataset runs and model saving

- `--seeding`: Turn seeding on or off. Default is True.
- `--seed`: Random seed. Default is 0.
- `--multiple_executions`: Turn multiple runs on or off. Default is True.
- `--runs`: Number of times the script runs. Default is 5.
- `--all_datasets`: Run the script for all 5 datasets at once or not. Default is True.
- `--dataset_name`: Choose which dataset to train for. Default is eth.
- `--check_testset`: Also evaluate on the testset, next to validation set. Default is True.
- `--output_dir`: Path where models are saved. Default is current directory.
- `--save_model_name_ADE`: Name of the saved model with best ADE. Default is saved_model_ADE.
- `--save_model_name_FDE`: Name of the saved model with best FDE. Default is saved_model_FDE.
- `--num_samples_check`: Limit the number of samples during metric calculation. Default is 5000.
- `--check_validation_every`: Check the metrics on the validation dataset every X epochs. Default is 1.


### Additional settings

- `--obs_len`: How many timesteps used for observation. Default is 8.
- `--pred_len`: How many timesteps used for prediction. Default is 12.
- `--discriminator_steps`: How many discriminator updates per iteration. Default is 1.
- `--policy_steps`: How many policy updates per iteration. Default is 1.
- `--loader_num_workers`: Number cpu/gpu processes. Default is 0.
- `--skip`: Number of frames to skip while making the dataset. Default is 1.
- `--delim`: How to read the data text file spacing. Default is \t.
- `--l2_loss_weight`: L2 loss multiplier. Default is 0.
- `--use_gpu`: How many GPUs to use (0 is cpu only). Default is 1.
- `--load_saved_model`: Path of pre-trained model. Default is None (don't use pretrained model).



## Running Models
You can use the script `scripts/evaluate_model.py` to easily run any of the pretrained models on any of the datsets.
Copying your models to /models/irl-models, in combination with the correct settings will compute the metrics for each model individually.
Copying your models to /models/model_average will give you the averaged results.
The evaluation script can be configured with the following settings:

- `--dataset_name`: Choose which dataset to evaluate for. Default is eth.
- `--model_path`: Directory of the models. Default is "../models/irl-models".
- `--num_samples`: Best-of-k for mADE/mFDE metrics. Default is 20.
- `--dset_type`: Which dataset split to evaluate on. Default is test.
- `--model_average`: Compute the average of the models or not. Default is True.
- `--runs`: Number of models to take the average of. Default is 5.
- `--prediction_steps`: Number of future trajectory points to predict. Default is 12.
- `--noise`: Add noise to the input or not. Default is False.




