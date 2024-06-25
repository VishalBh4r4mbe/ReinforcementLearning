from absl import app
from absl import flags
from absl import logging
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


import multiprocessing
import numpy as np
import torch
import copy

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'Pong',
    'The name of the environment'
)
flags.DEFINE_integer(
    'environment_height',
    84,
    'The height of the environment'
)
flags.DEFINE_integer(
    'environment_width',
    84,
    'The width of the environment'
)
flags.DEFINE_integer(
    'environment_frame_skip',
    4,
    'The number of frames to skip'
)
flags.DEFINE_integer(
    'environment_frame_stack',
    1,
    'The number of frames to stack'
)
flags.DEFINE_integer(
    'num_actors',
    8,
    'The number of actors'
)
flags.DEFINE_bool(
    'compress_state',
    True,
    'Compress state images when stored'
)
flags.DEFINE_integer(
    'replay_size',
    20000,
    'maximum replay size'
)
flags.DEFINE_integer(
    'min_replay_size',
    1000,
    'Minimum replay size before learning'
)
flags.DEFINE_bool(
    'clip_gradient',
    True,
    'Clip gradients'
)
flags.DEFINE_bool(
    'max_gradient_norm',
    40.0,
    'max gradient norm'
)
flags.DEFINE_float(
    'learning_rate',
    0.0001,
    'learning rate'
)
flags.DEFINE_float(
    'intrinsic_learning_rate',
    0.0005,
    'intrinsic learning rate[For embedding and RND]'
)
flags.DEFINE_float('extrinsic_discount', 0.997, 'Extrinsic reward discount rate.')
flags.DEFINE_float('intrinsic_discount', 0.99, 'Intrinsic reward discount rate.')
flags.DEFINE_float('adam_epsilon', 0.0001, 'Epsilon for adam.')
flags.DEFINE_integer('unroll_length', 80, 'Sequence of transitions to unroll before add to replay.')
flags.DEFINE_integer(
    'burn_in',
    40,
    'Sequence of transitions used to pass RNN before actual learning.'
    'The effective length of unrolls will be burn_in + unroll_length, '
    'two consecutive unrolls will overlap on burn_in steps.',
)
flags.DEFINE_integer('batch_size', 32, 'Batch size for learning.')

flags.DEFINE_float('policy_beta', 0.3, 'Scalar for the intrinsic reward scale.')
flags.DEFINE_integer('num_policies', 32, 'Number of directed policies to learn, scaled by intrinsic reward scale beta.')

flags.DEFINE_integer('episodic_memory_size', 30000, 'Maximum size of episodic memory.')  # 30000
flags.DEFINE_bool(
    'reset_episodic_memory',
    True,
    'Reset the episodic_memory on every episode, only applicable to actors, default on.'
    'From NGU Paper on MontezumaRevenge, Instead of resetting the memory after every episode, we do it after a small number of '
    'consecutive episodes, which we call a meta-episode. This structure plays an important role when the'
    'agent faces irreversible choices.',
)
flags.DEFINE_integer('num_neighbors', 10, 'Number of K-nearest neighbors.')
flags.DEFINE_float('kernel_epsilon', 0.0001, 'K-nearest neighbors kernel epsilon.')
flags.DEFINE_float('cluster_distance', 0.008, 'K-nearest neighbors custer distance.')
flags.DEFINE_float('max_similarity', 8.0, 'K-nearest neighbors custer distance.')

flags.DEFINE_float('retrace_lambda', 0.95, 'Lambda coefficient for retrace.')
flags.DEFINE_bool('transformed_retrace', True, 'Transformed retrace loss, default on.')

flags.DEFINE_float('priority_exponent', 0.9, 'Priority exponent used in prioritized replay.')
flags.DEFINE_float('importance_sampling_exponent', 0.6, 'Importance sampling exponent value.')
flags.DEFINE_bool('normalize_weights', True, 'Normalize sampling weights in prioritized replay.')
flags.DEFINE_float('priority_eta', 0.9, 'Priority eta to mix the max and mean absolute TD errors.')

flags.DEFINE_integer('num_iterations', 100, 'Number of iterations to run.')
flags.DEFINE_integer(
    'num_train_steps', int(5e5), 'Number of training steps (environment steps or frames) to run per iteration, per actor.'
)
flags.DEFINE_integer(
    'num_eval_steps', int(2e4), 'Number of evaluation steps (environment steps or frames) to run per iteration.'
)
flags.DEFINE_integer('max_episode_steps', 108000, 'Maximum steps (before frame skip) per episode.')
flags.DEFINE_integer(
    'target_net_update_interval',
    1500,
    'The interval (meassured in Q network updates) to update target Q networks.',
)
flags.DEFINE_integer('actor_update_interval', 100, 'The frequency (measured in actor steps) to update actor local Q network.')
flags.DEFINE_float('eval_exploration_epsilon', 0.0001, 'Fixed exploration rate in e-greedy policy for evaluation.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('use_tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_bool('actors_on_gpu', True, 'Run actors on GPU, default on.')
flags.DEFINE_integer(
    'debug_screenshots_interval',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', './logs/ngu_atari_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Path for checkpoint directory.')

flags.register_validator('environment_frame_stack', lambda x: x == 1)

def main(argv):
    del argv
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs NGU agent on {device}')
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    
    