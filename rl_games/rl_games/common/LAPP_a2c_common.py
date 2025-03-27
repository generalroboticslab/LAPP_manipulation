import copy
import os

from rl_games.common import vecenv

from rl_games.algos_torch.moving_mean_std import GeneralizedMovingStats
from rl_games.algos_torch.self_play_manager import SelfPlayManager
from rl_games.algos_torch import torch_ext
from rl_games.common import schedulers
from rl_games.common.experience import ExperienceBuffer
from rl_games.common.interval_summary_writer import IntervalSummaryWriter
from rl_games.common.diagnostics import DefaultDiagnostics, PpoDiagnostics
from rl_games.algos_torch import  model_builder
from rl_games.interfaces.base_algorithm import  BaseAlgorithm
import numpy as np
import time
import gym

from datetime import datetime
from tensorboardX import SummaryWriter
import torch 
from torch import nn
import torch.distributed as dist
 
from time import sleep

from rl_games.common import common_losses

# new import
from .LAPP_transformer import TransformerTrainer
import ast
import random
from openai import OpenAI
from pathlib import Path


def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


def print_statistics(print_stats, curr_frames, step_time, step_inference_time, total_time, epoch_num, max_epochs, frame, max_frames):
    if print_stats:
        step_time = max(step_time, 1e-9)
        fps_step = curr_frames / step_time
        fps_step_inference = curr_frames / step_inference_time
        fps_total = curr_frames / total_time

        if max_epochs == -1 and max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}')
        elif max_epochs == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}/{max_frames:.0f}')
        elif max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}')
        else:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}/{max_frames:.0f}')


class LAPPA2CBase(BaseAlgorithm):

    def __init__(self, base_name, params):

        self.config = config = params['config']
        pbt_str = ''
        self.population_based_training = config.get('population_based_training', False)
        if self.population_based_training:
            # in PBT, make sure experiment name contains a unique id of the policy within a population
            pbt_str = f'_pbt_{config["pbt_idx"]:02d}'

        # This helps in PBT when we need to restart an experiment with the exact same name, rather than
        # generating a new name with the timestamp every time.
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['log_dir']
            # self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")
            # self.experiment_name = config['name'] + pbt_str + '-{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())

        self.config = config
        self.algo_observer = config['features']['observer']
        self.algo_observer.before_init(base_name, config, self.experiment_name)
        self.load_networks(params)
        self.multi_gpu = config.get('multi_gpu', False)
        self.rank = 0
        self.rank_size = 1
        self.curr_frames = 0

        if self.multi_gpu:
            self.rank = int(os.getenv("LOCAL_RANK", "0"))
            self.rank_size = int(os.getenv("WORLD_SIZE", "1"))
            dist.init_process_group("nccl", rank=self.rank, world_size=self.rank_size)

            self.device_name = 'cuda:' + str(self.rank)
            config['device'] = self.device_name
            if self.rank != 0:
                config['print_stats'] = False
                config['lr_schedule'] = None

        self.use_diagnostics = config.get('use_diagnostics', False)

        if self.use_diagnostics and self.rank == 0:
            self.diagnostics = PpoDiagnostics()
        else:
            self.diagnostics = DefaultDiagnostics()

        self.network_path = config.get('network_path', "./nn/")
        self.log_path = config.get('log_path', "runs/")
        self.env_config = config.get('env_config', {})
        self.num_actors = config['num_actors']
        self.env_name = config['env_name']

        self.vec_env = None
        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()
        else:
            self.vec_env = config.get('vec_env', None)
        self.env = self.vec_env.env
        if self.env is not None:
            print("!!! self.env created successfully!")
        self.env.LAPP_init_buf()

        self.ppo_device = config.get('device', 'cuda:0')
        self.value_size = self.env_info.get('value_size',1)
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.central_value_config = self.config.get('central_value_config', None)
        self.has_central_value = self.central_value_config is not None
        self.truncate_grads = self.config.get('truncate_grads', False)

        if self.has_central_value:
            self.state_space = self.env_info.get('state_space', None)
            if isinstance(self.state_space,gym.spaces.Dict):
                self.state_shape = {}
                for k,v in self.state_space.spaces.items():
                    self.state_shape[k] = v.shape
            else:
                self.state_shape = self.state_space.shape

        self.self_play_config = self.config.get('self_play_config', None)
        self.has_self_play_config = self.self_play_config is not None

        self.self_play = config.get('self_play', False)
        self.save_freq = config.get('save_frequency', 0)
        self.save_best_after = config.get('save_best_after', 100)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name

        self.ppo = config.get('ppo', True)
        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)

        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.linear_lr = config['lr_schedule'] == 'linear'
        self.schedule_type = config.get('schedule_type', 'legacy')

        # Setting learning rate scheduler
        if self.is_adaptive_lr:
            self.kl_threshold = config['kl_threshold']
            self.scheduler = schedulers.AdaptiveScheduler(self.kl_threshold)

        elif self.linear_lr:
            
            if self.max_epochs == -1 and self.max_frames == -1:
                print("Max epochs and max frames are not set. Linear learning rate schedule can't be used, switching to the contstant (identity) one.")
                self.scheduler = schedulers.IdentityScheduler()
            else:
                use_epochs = True
                max_steps = self.max_epochs

                if self.max_epochs == -1:
                    use_epochs = False
                    max_steps = self.max_frames

                self.scheduler = schedulers.LinearScheduler(float(config['learning_rate']), 
                    max_steps = max_steps,
                    use_epochs = use_epochs, 
                    apply_to_entropy = config.get('schedule_entropy', False),
                    start_entropy_coef = config.get('entropy_coef'))
        else:
            self.scheduler = schedulers.IdentityScheduler()

        self.e_clip = config['e_clip']
        self.clip_value = config['clip_value']
        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.num_agents = self.env_info.get('agents', 1)
        self.horizon_length = config['horizon_length']
        self.seq_len = self.config.get('seq_length', 4)
        self.bptt_len = self.config.get('bptt_length', self.seq_len) # not used right now. Didn't show that it is usefull
        self.zero_rnn_on_done = self.config.get('zero_rnn_on_done', True)
        self.normalize_advantage = config['normalize_advantage']
        self.normalize_rms_advantage = config.get('normalize_rms_advantage', False)
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        self.truncate_grads = self.config.get('truncate_grads', False)

        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k,v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
 
        self.critic_coef = config['critic_coef']
        self.grad_norm = config['grad_norm']
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']

        self.games_to_track = self.config.get('games_to_track', 100)
        print('current training device:', self.ppo_device)
        self.game_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)

        self.pref_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)

        self.game_shaped_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
        self.obs = None
        self.games_num = self.config['minibatch_size'] // self.seq_len # it is used only for current rnn implementation
        self.batch_size = self.horizon_length * self.num_actors * self.num_agents
        self.batch_size_envs = self.horizon_length * self.num_actors
        assert(('minibatch_size_per_env' in self.config) or ('minibatch_size' in self.config))
        self.minibatch_size_per_env = self.config.get('minibatch_size_per_env', 0)
        self.minibatch_size = self.config.get('minibatch_size', self.num_actors * self.minibatch_size_per_env)
        self.mini_epochs_num = self.config['mini_epochs']

        # Jason (7/10): automatically scale down minibatch size if it is too large
        if self.batch_size < self.minibatch_size:
            self.minibatch_size = self.batch_size

        self.num_minibatches = self.batch_size // self.minibatch_size
        assert(self.batch_size % self.minibatch_size == 0)

        self.mixed_precision = self.config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.last_lr = self.config['learning_rate']
        self.frame = 0
        self.update_time = 0
        self.mean_rewards = self.last_mean_rewards = -100500
        self.last_mean_successes = -1.0
        self.play_time = 0
        self.epoch_num = 0
        self.curr_frames = 0
        # allows us to specify a folder where all experiments will reside
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        # self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        # self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        # self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')
        self.nn_dir = os.path.join(self.train_dir, 'nn')
        self.summaries_dir = os.path.join(self.train_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        # os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.entropy_coef = self.config['entropy_coef']

        if self.rank == 0:
            writer = SummaryWriter(self.summaries_dir)
            if self.population_based_training:
                self.writer = IntervalSummaryWriter(writer, self.config)
            else:
                self.writer = writer
        else:
            self.writer = None

        self.value_bootstrap = self.config.get('value_bootstrap')
        self.use_smooth_clamp = self.config.get('use_smooth_clamp', False)

        if self.use_smooth_clamp:
            self.actor_loss_func = common_losses.smoothed_actor_loss
        else:
            self.actor_loss_func = common_losses.actor_loss

        if self.normalize_advantage and self.normalize_rms_advantage:
            momentum = self.config.get('adv_rms_momentum', 0.5) #'0.25'
            self.advantage_mean_std = GeneralizedMovingStats((1,), momentum=momentum).to(self.ppo_device)

        self.is_tensor_obses = False

        self.last_rnn_indices = None
        self.last_state_indices = None

        # self_play
        if self.has_self_play_config:
            print('Initializing SelfPlay Manager')
            self.self_play_manager = SelfPlayManager(self.self_play_config, self.writer)

        # features
        self.algo_observer = config['features']['observer']

        self.soft_aug = config['features'].get('soft_augmentation', None)
        self.has_soft_aug = self.soft_aug is not None
        # soft augmentation not yet supported
        assert not self.has_soft_aug


        self.num_tot_pairs = 600
        self.num_collection = 100
        # region [DIY]
        '''
        Prefix "tot" refers to the full-length buffer of length 600; and prefix "tmp" refers to temporary buffer of length 100;
        We are updating the trajectory buffer every 100 epochs
        The parameter of num_tot_pairs & num_collection can be changed
        '''
        if self.config['task_name'] == 'hand_over':
            self.tot_object_pos_buf = torch.zeros(self.num_tot_pairs, 2, self.horizon_length, len(self.env.object_pos[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False) # the position of the object
            self.tot_object_linvel_buf = torch.zeros(self.num_tot_pairs, 2, self.horizon_length, len(self.env.object_linvel[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False) # the linear velocity of the object
            self.tot_success_buf = torch.zeros(self.num_tot_pairs, 2, self.horizon_length, 1,
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False) # the indicator of current epoch's success
            self.tot_dist_to_tip_buf = torch.zeros(self.num_tot_pairs, 2, self.horizon_length, 5,
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False) # distance of object to left hand's tips
            self.tot_dist_to_another_tip_buf = torch.zeros(self.num_tot_pairs, 2, self.horizon_length, 5,
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False) # distance of object to right hand's tips
        
        elif self.config['task_name'] == 'swing_cup':
            self.tot_object_linvel_buf = torch.zeros(self.num_tot_pairs, 2, self.horizon_length, len(self.env.object_linvel[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False) # same as above
            self.tot_object_rot_buf = torch.zeros(self.num_tot_pairs, 2, self.horizon_length, len(self.env.object_rot3[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False) # the rotation of the object
            self.tot_success_buf = torch.zeros(self.num_tot_pairs, 2, self.horizon_length, 1,
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False) # same as above
            self.tot_left_dist_buf = torch.zeros(self.num_tot_pairs, 2, self.horizon_length, 1,
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False) # distance between left handle and left hand
            self.tot_right_dist_buf = torch.zeros(self.num_tot_pairs, 2, self.horizon_length, 1,
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False) # distance between right handle and right hand
        
        elif self.config['task_name'] == 'kettle':
            self.tot_kettle_spout_pos_buf = torch.zeros(self.num_tot_pairs, 2, self.horizon_length, len(self.env.kettle_spout_pos[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False) # the position of kettle spout
            self.tot_bucket_handle_pos_buf = torch.zeros(self.num_tot_pairs, 2, self.horizon_length, len(self.env.bucket_handle_pos[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False) # the position of bucket handle
            self.tot_left_hand_ff_pos_buf = torch.zeros(self.num_tot_pairs, 2, self.horizon_length, len(self.env.left_hand_ff_pos[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False) # the position of left hand fore finger
            self.tot_right_hand_ff_pos_buf = torch.zeros(self.num_tot_pairs, 2, self.horizon_length, len(self.env.right_hand_ff_pos[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False) # the position of right hand fore finger
            self.tot_success_buf = torch.zeros(self.num_tot_pairs, 2, self.horizon_length, 1,
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False) # same as above
            self.tot_kettle_handle_pos_buf = torch.zeros(self.num_tot_pairs, 2, self.horizon_length, len(self.env.kettle_handle_pos[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False) # the position of kettle handle

        # ===========================================
        if self.config['task_name'] == 'hand_over':
            self.tmp_object_pos_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.object_pos[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False)
            self.tmp_object_linvel_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.object_linvel[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False)
            self.tmp_success_buf = torch.zeros(self.num_collection, 2, self.horizon_length, 1,
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False)
            self.tmp_dist_to_tip_buf = torch.zeros(self.num_collection, 2, self.horizon_length, 5,
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False)
            self.tmp_dist_to_another_tip_buf = torch.zeros(self.num_collection, 2, self.horizon_length, 5,
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False)
            
        elif self.config['task_name'] == 'swing_cup':
            self.tmp_object_linvel_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.object_linvel[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False)
            self.tmp_object_rot_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.object_rot3[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False)
            self.tmp_success_buf = torch.zeros(self.num_collection, 2, self.horizon_length, 1,
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False)
            self.tmp_left_dist_buf = torch.zeros(self.num_collection, 2, self.horizon_length, 1,
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False)
            self.tmp_right_dist_buf = torch.zeros(self.num_collection, 2, self.horizon_length, 1,
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False)
            
        elif self.config['task_name'] == 'kettle':
            self.tmp_kettle_spout_pos_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.kettle_spout_pos[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False)
            self.tmp_bucket_handle_pos_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.bucket_handle_pos[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False)
            self.tmp_left_hand_ff_pos_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.left_hand_ff_pos[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False)
            self.tmp_right_hand_ff_pos_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.right_hand_ff_pos[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False)
            self.tmp_success_buf = torch.zeros(self.num_collection, 2, self.horizon_length, 1,
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False)
            self.tmp_kettle_handle_pos_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.kettle_handle_pos[0]),
                                                device=self.ppo_device, dtype=torch.float, requires_grad=False)
        # endregion

        self.pref_load_error_value = 4 # to denote slot where there's no preference label
        self.pref_scale = config['pref_scale']

        # llm part
        '''
        The following is the code of configuring the LLM.
        '''
        self.current_file_path = Path(__file__).resolve()
        self.project_root = self.current_file_path.parents[3]
        # region [TODO] put in your api key
        if config['key_path'] == 'personal':
            self.gpt_key_file_path = self.project_root / 'api_key/personal_api_key.txt'
        # endregion
        else:
            self.gpt_key_file_path = self.project_root / 'api_key/empty.txt'
        self.OPENAI_API_KEY = self.get_secret_key()
        self.client = OpenAI(api_key=self.OPENAI_API_KEY)

        self.model_name = 'gpt-4o-mini'  # gpt-4o-mini, gpt-4o-2024-11-20
        # print("Current model is", self.model_name)
        self.temperature = 1.0
        self.n_samples = 15
        self.num_pair_per_prompt = 5

        self.track_price = 0

        # region [DIY]
        '''
        This part is to load the prompt template. If you are designing your own, please properly put the path.
        '''
        if self.config['task_name'] == 'hand_over':
            self.init_sys_file_path = self.project_root / 'LAPP_prompt_library/shadow_hand_over/hand_over_init_sys.txt'
        elif self.config['task_name'] == 'swing_cup':
            self.init_sys_file_path = self.project_root / 'LAPP_prompt_library/shadow_hand_swing_cup/swing_cup_init_sys.txt'
        elif self.config['task_name'] == 'kettle':
            self.init_sys_file_path = self.project_root / 'LAPP_prompt_library/shadow_hand_kettle/kettle_init_sys.txt'
        #endregion

        self.initial_system = self.file2str(self.init_sys_file_path)

        self.pref_predictor = None
        self.pref_train_data = None
        self.buf_interval = 100

        self.cur_ep_start_id = 0

    def trancate_gradients_and_step(self):
        if self.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            all_grads_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))
            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.rank_size
                    )
                    offset += param.numel()

        if self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)
        has_central_value_net = self.config.get('central_value_config') is not  None
        if has_central_value_net:
            print('Adding Central Value Network')
            if 'model' not in params['config']['central_value_config']:
                params['config']['central_value_config']['model'] = {'name': 'central_value'}
            network = builder.load(params['config']['central_value_config'])
            self.config['central_value_config']['network'] = network

    def write_stats(self, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames):
        # do we need scaled time?
        self.diagnostics.send_info(self.writer)
        self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
        self.writer.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
        self.writer.add_scalar('performance/step_fps', curr_frames / step_time, frame)
        self.writer.add_scalar('performance/rl_update_time', update_time, frame)
        self.writer.add_scalar('performance/step_inference_time', play_time, frame)
        self.writer.add_scalar('performance/step_time', step_time, frame)
        self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(a_losses).item(), frame)
        self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(c_losses).item(), frame)
                
        self.writer.add_scalar('losses/entropy', torch_ext.mean_list(entropies).item(), frame)
        self.writer.add_scalar('info/last_lr', last_lr * lr_mul, frame)
        self.writer.add_scalar('info/lr_mul', lr_mul, frame)
        self.writer.add_scalar('info/e_clip', self.e_clip * lr_mul, frame)
        self.writer.add_scalar('info/kl', torch_ext.mean_list(kls).item(), frame)
        self.writer.add_scalar('info/epochs', epoch_num, frame)
        self.algo_observer.after_print_stats(frame, epoch_num, total_time)

    def set_eval(self):
        self.model.eval()
        if self.normalize_rms_advantage:
            self.advantage_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_rms_advantage:
            self.advantage_mean_std.train()

    def update_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        #if self.has_central_value:
        #    self.central_value_net.update_lr(lr)

    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        return res_dict

    def get_values(self, obs):
        with torch.no_grad():
            if self.has_central_value:
                states = obs['states']
                self.central_value_net.eval()
                input_dict = {
                    'is_train': False,
                    'states' : states,
                    'actions' : None,
                    'is_done': self.dones,
                }
                value = self.get_central_value(input_dict)
            else:
                self.model.eval()
                processed_obs = self._preproc_obs(obs['obs'])
                input_dict = {
                    'is_train': False,
                    'prev_actions': None, 
                    'obs' : processed_obs,
                    'rnn_states' : self.rnn_states
                }
                result = self.model(input_dict)
                value = result['values']
            return value

    @property
    def device(self):
        return self.ppo_device

    def reset_envs(self):
        self.obs = self.env_reset()

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }
        self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.ppo_device)

        val_shape = (self.horizon_length, batch_size, self.value_size)
        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_shaped_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)
        self.cur_pref_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)

        if self.is_rnn:
            self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]

            total_agents = self.num_agents * self.num_actors
            num_seqs = self.horizon_length // self.seq_len
            assert((self.horizon_length * total_agents // self.num_minibatches) % self.seq_len == 0)
            self.mb_rnn_states = [torch.zeros((num_seqs, s.size()[0], total_agents, s.size()[2]), dtype = torch.float32, device=self.ppo_device) for s in self.rnn_states]

    def init_rnn_from_model(self, model):
        self.is_rnn = self.model.is_rnn()

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.ppo_device)
            else:
                obs = torch.FloatTensor(obs).to(self.ppo_device)
        return obs

    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:    
            upd_obs = {'obs' : upd_obs}
        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos

    def env_reset(self):
        obs = self.vec_env.reset()
        obs = self.obs_to_tensors(obs)
        return obs

    def discount_values(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
        return mb_advs

    def discount_values_masks(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards, mb_masks):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)
        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)
            masks_t = mb_masks[t].unsqueeze(1)
            delta = (mb_rewards[t] + self.gamma * nextvalues * nextnonterminal  - mb_extrinsic_values[t])
            mb_advs[t] = lastgaelam = (delta + self.gamma * self.tau * nextnonterminal * lastgaelam) * masks_t
        return mb_advs

    def clear_stats(self):
        batch_size = self.num_agents * self.num_actors
        self.game_rewards.clear()
        self.game_shaped_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -100500
        self.algo_observer.after_clear_stats()

    def update_epoch(self):
        pass

    def train(self):
        pass

    def prepare_dataset(self, batch_dict):
        pass

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

    def train_actor_critic(self, obs_dict, opt_step=True):
        pass

    def calc_gradients(self):
        pass

    def get_central_value(self, obs_dict):
        return self.central_value_net.get_value(obs_dict)

    def train_central_value(self):
        return self.central_value_net.train_net()

    def get_full_state_weights(self):
        state = self.get_weights()
        state['epoch'] = self.epoch_num
        state['optimizer'] = self.optimizer.state_dict()
        if self.has_central_value:
            state['assymetric_vf_nets'] = self.central_value_net.state_dict()
        state['frame'] = self.frame

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state['last_mean_rewards'] = self.last_mean_rewards

        if self.vec_env is not None:
            env_state = self.vec_env.get_env_state()
            state['env_state'] = env_state

        return state

    def set_full_state_weights(self, weights):
        self.set_weights(weights)
        if self.has_central_value:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])

        self.optimizer.load_state_dict(weights['optimizer'])
        self.frame = weights.get('frame', 0)
        self.last_mean_rewards = weights.get('last_mean_rewards', -100500)

        env_state = weights.get('env_state', None)

        if self.vec_env is not None:
            self.vec_env.set_env_state(env_state)

    def get_weights(self):
        state = self.get_stats_weights()
        state['model'] = self.model.state_dict()
        return state

    def get_stats_weights(self, model_stats=False):
        state = {}
        if self.mixed_precision:
            state['scaler'] = self.scaler.state_dict()
        if self.has_central_value:
            state['central_val_stats'] = self.central_value_net.get_stats_weights(model_stats)
        if model_stats:
            if self.normalize_input:
                state['running_mean_std'] = self.model.running_mean_std.state_dict()
            if self.normalize_value:
                state['reward_mean_std'] = self.model.value_mean_std.state_dict()

        return state

    def set_stats_weights(self, weights):
        if self.normalize_rms_advantage:
            self.advantage_mean_std.load_state_dic(weights['advantage_mean_std'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])
        if self.normalize_value and 'normalize_value' in weights:
            self.model.value_mean_std.load_state_dict(weights['reward_mean_std'])
        if self.mixed_precision and 'scaler' in weights:
            self.scaler.load_state_dict(weights['scaler'])

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        self.set_stats_weights(weights)

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k,v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch


    def update_train_data_queue(self, new_data, train_data=None):
        '''
        Update the training data queue by appending new data and removing the oldest entries.
        Input variables:
            new_data: dict - new training data to be added.
            train_data: dict - existing training data queue.
        Output:
            dict - the updated training data queue.
        '''
        if train_data is None:
            train_data = self.pref_train_data
        assert set(train_data.keys()) == set(new_data.keys()), "Keys in old 500 and new 100 data must match"
        new_data_length = len(new_data['pref_label_buf'])

        for key in train_data.keys():
            train_data[key] = torch.cat(
                (train_data[key][new_data_length:], new_data[key]),
                dim=0
            )
        
        return train_data

    def select_rand_pairs(self, num_traj, num_pairs):
        '''
        Randomly select a specified number of unique trajectory pairs.
        Input variables:
            num_traj: int - total number of trajectories available.
            num_pairs: int - number of unique pairs to select.
        Output:
            numpy.ndarray - an array of selected unique trajectory pairs.
        '''
        if num_pairs > num_traj * (num_traj-1) // 2:
            raise ValueError("Number of pairs requested exceed total possible unique pairs.")
        
        pairs = set()

        while len(pairs) < num_pairs:
            i = np.random.randint(0, num_traj)
            j = np.random.randint(0, num_traj)

            if i != j:
                pair = (min(i, j), max(i, j))
                pairs.add(pair)

        pairs_array = np.array(list(pairs))
        return pairs_array
    

    def compute_col_mode(self, pref_val_tensor_pool: torch.Tensor):
        '''
        Compute the mode (most frequent value) for each column in the given tensor.
        If there are ties, a random value among the most frequent is chosen.
        Input variables:
            pref_val_tensor_pool: torch.Tensor - tensor containing preference values.
        Output:
            torch.Tensor - tensor of mode values for each column.
        '''
        values, _ = torch.mode(pref_val_tensor_pool, dim=0)

        for col in range(pref_val_tensor_pool.size(1)):
            col_val, col_cnt = torch.unique(pref_val_tensor_pool[:, col], return_counts=True)

            max_cnt = col_cnt.max()
            candidates = col_val[col_cnt == max_cnt]

            if candidates.numel() > 1:
                rand_idx = torch.randint(0, candidates.numel(), (1,))
                values[col] = candidates[rand_idx]

        return values
    
    
    def create_mixed_data(self, input_data: dict, preserve_frac: float = 0.6, seed: int = None) -> dict:
        '''
        Create a new mixed dataset by preserving a fraction of the original data and generating new trajectory pairs.
        Input variables:
            input_data: dict - original dataset containing tensors.
            preserve_frac: float - fraction of data to preserve.
            seed: int - seed for random number generation.
        Output:
            dict - the new mixed dataset.
        '''
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # check dimension
        all_keys = list(input_data.keys())
        if len(all_keys) == 0:
            raise ValueError("input_data cannot be an empty dict.")
        
        first_key = all_keys[0]
        first_shape = input_data[first_key].shape
        if len(first_shape) != 4:
            raise ValueError(f"All tensors must have 4 dimensions, got shape {first_shape} for key {first_key}.")
        num_pair, two_, traj_len, _ = first_shape
        if two_ != 2:
            raise ValueError(f"The second dimension must be 2, but got {two_}.")
        
        for k in all_keys[1:]:
            shape_k = input_data[k].shape
            if len(shape_k) != 4:
                raise ValueError(f"Key {k} has shape {shape_k}; all must have 4 dims.")
            if shape_k[0] != num_pair or shape_k[1] != 2 or shape_k[2] != traj_len:
                raise ValueError(
                    f"Key {k} has shape {shape_k}, but expected (num_pair={num_pair}, 2, traj_len={traj_len}, ...)."
                )
            
        num_to_preserve = int(round(preserve_frac * num_pair))
        num_new = num_pair - num_to_preserve

        # Create new dataset
        flattened_data = {}
        for k in all_keys:
            pk, two, tl, fdim = input_data[k].shape
            flattened_data[k] = input_data[k].view(pk * 2, tl, fdim)

        new_data = {}
        for k in all_keys:
            new_data[k] = torch.empty_like(input_data[k])

        all_pair_idx = list(range(num_pair))
        preserve_idx = random.sample(all_pair_idx, num_to_preserve)

        for out_idx, old_pair_idx in enumerate(preserve_idx):
            for k in all_keys:
                new_data[k][out_idx] = input_data[k][old_pair_idx]

        old_pairs = set((2 * i, 2 * i + 1) for i in range(num_pair))

        all_traj_indices = list(range(2 * num_pair))
        new_pairs = []

        while len(new_pairs) < num_new:
            i, j = random.sample(all_traj_indices, 2)
            pair_tuple = tuple(sorted((i, j)))
            if pair_tuple not in old_pairs and pair_tuple not in new_pairs:
                new_pairs.append(pair_tuple)

        for idx, (i, j) in enumerate(new_pairs):
            out_idx = num_to_preserve + idx
            for k in all_keys:
                traj_i = flattened_data[k][i]
                traj_j = flattened_data[k][j]
                new_data[k][out_idx] = torch.stack([traj_i, traj_j], dim=0)

        return new_data

    def pref_val_str2tensor(self, pref_str: str, prompt_batchsize: int, device):
        '''
        Convert a string representation of preference values to a tensor.
        Returns a default error tensor if conversion fails.
        Input variables:
            pref_str: str - string containing the preference values.
            prompt_batchsize: int - expected number of preference values.
            device: int - cuda device to create the tensor on.
        Output:
            torch.Tensor - tensor of preference values.
        '''
        try:
            pref_values = ast.literal_eval(pref_str)

            if not isinstance(pref_values, list):
                raise ValueError("The response is not a list.")
            
            if len(pref_values) != prompt_batchsize:
                raise ValueError(f"The reponse does not contain exactly {prompt_batchsize} elements.")

            valid_numbers = {0,1,2,3}
            for num in pref_values:
                if not isinstance(num, int):
                    raise TypeError(f"Invalid type {type(num)}.")
                if num not in valid_numbers:
                    raise KeyError(f"Invalie value {num}.")
                
            pref_values_tensor = torch.tensor(pref_values, dtype=torch.int, device=device)
            return pref_values_tensor
        
        except Exception as e:
            print(f"Invalid GPT response at index: {pref_str}")
            print(f"Error: {e}")
            pref_values_tensor = torch.zeros(prompt_batchsize, dtype=torch.int, device=device)
            pref_values_tensor.fill_(self.pref_load_error_value)
            return pref_values_tensor
        

    def get_secret_key(self):
        '''
        Retrieve the secret API key from the specified file.
        Output:
            str - the secret API key or an error message if not found.
        '''
        try:
            with open(self.gpt_key_file_path, 'r') as file:
                apikey = file.read().strip()
            return apikey
        except FileNotFoundError:
            return "!!! API key file is not found."
        except Exception as e:
            return f"An error occured: {e}"
        

    def file2str(self, file_path):
        with open(file_path, 'r', errors="ignore") as file:
            return file.read()
        

    def gpt_gen_pref_labels(self, data: dict):
        '''
        Generate preference labels using GPT based on provided trajectory data.
        The function processes data, calls GPT for responses, and updates the data dictionary.
        Input variables:
            data: dict - trajectory data for which preference labels need to be generated.
        Output:
            dict - updated data dictionary with a new 'pref_label_buf' entry.
        '''
        # extract data
        # region [DIY]
        if self.config['task_name'] == 'hand_over':
            object_pos_buf = data['object_pos_buf'].cpu().numpy()
            object_linvel_buf = data['object_linvel_buf'].cpu().numpy()
            success_buf = data['success_buf'].cpu().numpy()
            dist_buf = data['dist_buf'].cpu().numpy()
            dist_another_buf = data['dist_another_buf'].cpu().numpy()

        elif self.config['task_name'] == 'swing_cup':
            object_linvel_buf = data['object_linvel_buf'].cpu().numpy()
            object_rot_buf = data['object_rot_buf'].cpu().numpy()
            left_dist_buf = data['left_dist_buf'].cpu().numpy()
            right_dist_buf = data['right_dist_buf'].cpu().numpy()
            success_buf = data['success_buf'].cpu().numpy()

        elif self.config['task_name'] == 'kettle':
            kettle_spout_pos_buf = data['kettle_spout_pos_buf'].cpu().numpy()
            kettle_handle_pos_buf = data['kettle_handle_pos_buf'].cpu().numpy()
            bucket_handle_pos_buf = data['bucket_handle_pos_buf'].cpu().numpy()
            left_hand_ff_pos_buf = data['left_hand_ff_pos_buf'].cpu().numpy()
            right_hand_ff_pos_buf = data['right_hand_ff_pos_buf'].cpu().numpy()
            success_buf = data['success_buf'].cpu().numpy()
        # endregion

        # claim empty label buffer
        pref_label_buf = torch.zeros(len(success_buf), dtype=torch.int, device=self.ppo_device, requires_grad=False)
        pref_label_buf.fill_(self.pref_load_error_value)

        # prepare conversation
        conversation_history = []
        assert len(success_buf) % self.num_pair_per_prompt == 0
        num_prompt = len(success_buf) // self.num_pair_per_prompt

        float_formatter = lambda x: f"{x:.3f}"
        int_formatter = lambda x: f"{int(x)}"
        dummy_large_number = 1_000_000

        for prompt_id in range(num_prompt):
            conversation_history = [{"role": "system", 
                                     "content": self.initial_system}]
            user_chat = ''

            for pair_id in range(self.num_pair_per_prompt):
                user_chat += f'Here is trajectories pair {pair_id}:\n'
                for id_in_pair in range(2):
                    user_chat += f'For trajectory {id_in_pair} in trajectories pair {pair_id}:\n'

                    # region [DIY]
                    if self.config['task_name'] == 'hand_over':
                        # [obj] pos
                        user_chat += 'The "object position" in this trajectory are:\n'
                        user_chat += np.array2string(object_pos_buf[prompt_id * self.num_pair_per_prompt + pair_id][id_in_pair],
                                                    formatter={'float_kind': float_formatter}, separator=' ',
                                                    threshold=dummy_large_number, max_line_width=dummy_large_number)
                        user_chat += '\n'
                        # [obj] lin vel
                        user_chat += 'The "object linear velocity" in this trajectory are:\n'
                        user_chat += np.array2string(object_linvel_buf[prompt_id * self.num_pair_per_prompt + pair_id][id_in_pair],
                                                    formatter={'float_kind': float_formatter}, separator=' ',
                                                    threshold=dummy_large_number, max_line_width=dummy_large_number)
                        user_chat += '\n'
                        # [hand0] object distance to fingertips
                        user_chat += 'The "distance to first hand fingertips" in this trajectory are:\n'
                        user_chat += np.array2string(dist_buf[prompt_id * self.num_pair_per_prompt + pair_id][id_in_pair],
                                                    formatter={'float_kind': float_formatter}, separator=' ',
                                                    threshold=dummy_large_number, max_line_width=dummy_large_number)
                        user_chat += '\n'
                        # [hand1] object distance to fingertips
                        user_chat += 'The "distance to second hand fingertips" in this trajectory are:\n'
                        user_chat += np.array2string(dist_another_buf[prompt_id * self.num_pair_per_prompt + pair_id][id_in_pair],
                                                    formatter={'float_kind': float_formatter}, separator=' ',
                                                    threshold=dummy_large_number, max_line_width=dummy_large_number)
                        user_chat += '\n'
                        # success
                        user_chat += 'The "success indicator" in this trajectory are:\n'
                        user_chat += np.array2string(np.squeeze(success_buf[prompt_id * self.num_pair_per_prompt + pair_id][id_in_pair]),
                                                    formatter={'all': int_formatter}, separator=' ',
                                                    threshold=dummy_large_number, max_line_width=dummy_large_number)
                        user_chat += '\n'
                    
                    elif self.config['task_name'] == 'swing_cup':
                        # [obj] lin vel
                        user_chat += 'The "object linear velocity" in this trajectory are:\n'
                        user_chat += np.array2string(object_linvel_buf[prompt_id * self.num_pair_per_prompt + pair_id][id_in_pair],
                                                    formatter={'float_kind': float_formatter}, separator=' ',
                                                    threshold=dummy_large_number, max_line_width=dummy_large_number)
                        user_chat += '\n'
                        # [obj] rot
                        user_chat += 'The "object angular orientation" in this trajectory are:\n'
                        user_chat += np.array2string(object_rot_buf[prompt_id * self.num_pair_per_prompt + pair_id][id_in_pair],
                                                    formatter={'float_kind': float_formatter}, separator=' ',
                                                    threshold=dummy_large_number, max_line_width=dummy_large_number)
                        user_chat += '\n'
                        # [dist] left
                        user_chat += 'The "left hand distance to left handle" in this trajectory are:\n'
                        user_chat += np.array2string(np.squeeze(left_dist_buf[prompt_id * self.num_pair_per_prompt + pair_id][id_in_pair]),
                                                    formatter={'float_kind': float_formatter}, separator=' ',
                                                    threshold=dummy_large_number, max_line_width=dummy_large_number)
                        user_chat += '\n'
                        # [dist] right
                        user_chat += 'The "right hand distance to right handle" in this trajectory are:\n'
                        user_chat += np.array2string(np.squeeze(right_dist_buf[prompt_id * self.num_pair_per_prompt + pair_id][id_in_pair]),
                                                    formatter={'float_kind': float_formatter}, separator=' ',
                                                    threshold=dummy_large_number, max_line_width=dummy_large_number)
                        user_chat += '\n'
                        # success
                        user_chat += 'The "success indicator" in this trajectory are:\n'
                        user_chat += np.array2string(np.squeeze(success_buf[prompt_id * self.num_pair_per_prompt + pair_id][id_in_pair]),
                                                    formatter={'all': int_formatter}, separator=' ',
                                                    threshold=dummy_large_number, max_line_width=dummy_large_number)
                        user_chat += '\n'

                    elif self.config['task_name'] == 'kettle':
                        # [kettle spout] pos
                        user_chat += 'The "kettle spout position" in this trajectory are:\n'
                        user_chat += np.array2string(kettle_spout_pos_buf[prompt_id * self.num_pair_per_prompt + pair_id][id_in_pair],
                                                    formatter={'float_kind': float_formatter}, separator=' ',
                                                    threshold=dummy_large_number, max_line_width=dummy_large_number)
                        user_chat += '\n'
                        # [kettle handle] pos
                        user_chat += 'The "kettle handle position" in this trajectory are:\n'
                        user_chat += np.array2string(kettle_handle_pos_buf[prompt_id * self.num_pair_per_prompt + pair_id][id_in_pair],
                                                    formatter={'float_kind': float_formatter}, separator=' ',
                                                    threshold=dummy_large_number, max_line_width=dummy_large_number)
                        user_chat += '\n'
                        # [bucket handle] pos
                        user_chat += 'The "bucket position" in this trajectory are:\n'
                        user_chat += np.array2string(bucket_handle_pos_buf[prompt_id * self.num_pair_per_prompt + pair_id][id_in_pair],
                                                    formatter={'float_kind': float_formatter}, separator=' ',
                                                    threshold=dummy_large_number, max_line_width=dummy_large_number)
                        user_chat += '\n'
                        # [left ff] pos
                        user_chat += 'The "left fore finger position" in this trajectory are:\n'
                        user_chat += np.array2string(left_hand_ff_pos_buf[prompt_id * self.num_pair_per_prompt + pair_id][id_in_pair],
                                                    formatter={'float_kind': float_formatter}, separator=' ',
                                                    threshold=dummy_large_number, max_line_width=dummy_large_number)
                        user_chat += '\n'
                        # [right ff] pos
                        user_chat += 'The "right fore finger position" in this trajectory are:\n'
                        user_chat += np.array2string(right_hand_ff_pos_buf[prompt_id * self.num_pair_per_prompt + pair_id][id_in_pair],
                                                    formatter={'float_kind': float_formatter}, separator=' ',
                                                    threshold=dummy_large_number, max_line_width=dummy_large_number)
                        user_chat += '\n'
                        # success
                        user_chat += 'The "success indicator" in this trajectory are:\n'
                        user_chat += np.array2string(np.squeeze(success_buf[prompt_id * self.num_pair_per_prompt + pair_id][id_in_pair]),
                                                    formatter={'all': int_formatter}, separator=' ',
                                                    threshold=dummy_large_number, max_line_width=dummy_large_number)
                        user_chat += '\n'
                    # endregion
                    
            user_chat += 'Now please provide preference feedback on these 5 pairs of trajectories according to the instructions in the initial system prompt.\n'
            user_chat += 'Please give response with only one list of 5 preference values, e.g., [0,0,1,2,3]. Do not provide any other text such as your comments or thoughts. The preference value number can only be 0, 1, 2, or 3.'
            conversation_history.append({"role": "user", "content": user_chat})

            # used for checking the chat message
            # ====================
            # if prompt_id == 0:
            #     print(user_chat)
            # ====================

            gpt_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=conversation_history,
                temperature=self.temperature,
                n=self.n_samples
            )           
            gpt_replies = [choice.message.content for choice in gpt_response.choices]
            
            # used for tracking the costs
            # ====================
            prompt_tokens = gpt_response.usage.prompt_tokens
            completion_tokens = gpt_response.usage.completion_tokens
            input_cost = (prompt_tokens * 0.15) / 1_000_000
            output_cost = (completion_tokens * 0.6) / 1_000_000
            self.track_price += input_cost + output_cost
            print(f"$$$ Cumulative price is {self.track_price}")
            # ====================
            
            pref_val_tensor_pool = torch.zeros(self.n_samples, self.num_pair_per_prompt, 
                                                dtype=torch.int, device=pref_label_buf.device)
            
            for response_id in range(self.n_samples):
                pref_val_tensor = self.pref_val_str2tensor(pref_str=gpt_replies[response_id],
                                                        prompt_batchsize=self.num_pair_per_prompt,
                                                        device=pref_label_buf.device)
                pref_val_tensor_pool[response_id] = pref_val_tensor
                
            pref_val_tensor_mode = self.compute_col_mode(pref_val_tensor_pool)
            
            pref_label_buf[prompt_id*self.num_pair_per_prompt: (prompt_id+1)*self.num_pair_per_prompt] = pref_val_tensor_mode
            time.sleep(0.4)
            
        data['pref_label_buf'] = pref_label_buf
        print("The preference labels are: ")
        print(data['pref_label_buf'])
        return data
        
    def play_steps(self, epoch_num):
        update_list = self.update_list

        step_time = 0.0
        
        if epoch_num < 301:
            selected_pairs_ids = self.select_rand_pairs(num_traj=self.env.num_envs, num_pairs=2)
        else:
            selected_pairs_ids = self.select_rand_pairs(num_traj=self.env.num_envs, num_pairs=1)
        
        if epoch_num % 100 == 1 and epoch_num > 250: # train new predictor at beginning of epoch [n01], n>=3
            self.pref_predictor = None

            torch.set_grad_enabled(True)

            # region [DIY/Ablation]
            self.pref_predictor = TransformerTrainer(org_data=self.pref_train_data,
                                                     device=self.ppo_device,
                                                     save_models_path=self.train_dir + '/predictor' + f'/pred_models_{epoch_num}.pt',
                                                     save_models=True,
                                                     pool_models_num=9,
                                                     select_models_num=3,
                                                     input_mode=0,
                                                     batch_size=256,
                                                     transformer_embed_dim=64,
                                                     seq_length=1,
                                                     epsilon=0.1,
                                                     lr=9e-4,
                                                     weight_decay=1e-4,
                                                     epochs=100,
                                                     task_name=self.config['task_name']) # reminder that you may change the code in transformer trainer
            # endregion
            _, _, _, _ = self.pref_predictor.train()
            
        traj_pref_rew_buf = torch.zeros(self.horizon_length, dtype=torch.float, device=self.ppo_device)

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])

            mean_step_pref_rew = None
            if self.pref_predictor is not None:
                # region [DIY]
                if self.config['task_name'] == 'hand_over':
                    cur_state = torch.cat((
                        self.env.object_pos,
                        self.env.object_linvel,
                        self.env.dist_to_tip,
                        self.env.dist_to_another_tip
                    ), dim=1)
                elif self.config['task_name'] == 'swing_cup':
                    cur_state = torch.cat((
                        self.env.object_linvel,
                        self.env.object_rot3,
                        self.env.left_dist.unsqueeze(1),
                        self.env.right_dist.unsqueeze(1)
                    ), dim=1)
                elif self.config['task_name'] == 'kettle':
                    cur_state = torch.cat((
                        self.env.kettle_spout_pos,
                        self.env.kettle_handle_pos,
                        self.env.bucket_handle_pos,
                        self.env.left_hand_ff_pos,
                        self.env.right_hand_ff_pos
                    ), dim=1)
                # endregion

                pref_rewards = self.pref_predictor.predict_batch_reward(cur_state)
                mean_step_pref_rew = pref_rewards.mean()
                traj_pref_rew_buf[n] = mean_step_pref_rew.item()

            # Collect data into buffer
            # region [DIY]
            if self.pref_predictor is not None:
                for k, sel_pair_id in enumerate(selected_pairs_ids):
                    cur_pair_id = self.cur_ep_start_id + k
                    if self.config['task_name'] == 'hand_over':
                        for idx_in_pair in range(2):
                            self.tmp_object_pos_buf[cur_pair_id][idx_in_pair][n][:] = self.env.object_pos[sel_pair_id[idx_in_pair]][:]
                            self.tmp_object_linvel_buf[cur_pair_id][idx_in_pair][n][:] = self.env.object_linvel[sel_pair_id[idx_in_pair]][:]
                            self.tmp_success_buf[cur_pair_id][idx_in_pair][n][:] = 1 if infos['successes'][sel_pair_id[idx_in_pair]] != 0 else 0
                            self.tmp_dist_to_tip_buf[cur_pair_id][idx_in_pair][n][:] = self.env.dist_to_tip[sel_pair_id[idx_in_pair]][:]
                            self.tmp_dist_to_another_tip_buf[cur_pair_id][idx_in_pair][n][:] = self.env.dist_to_another_tip[sel_pair_id[idx_in_pair]][:]
                    elif self.config['task_name'] == 'swing_cup':
                        for idx_in_pair in range(2):
                            self.tmp_object_linvel_buf[cur_pair_id][idx_in_pair][n][:] = self.env.object_linvel[sel_pair_id[idx_in_pair]][:]
                            self.tmp_object_rot_buf[cur_pair_id][idx_in_pair][n][:] = self.env.object_rot3[sel_pair_id[idx_in_pair]][:]
                            self.tmp_left_dist_buf[cur_pair_id][idx_in_pair][n][:] = self.env.left_dist[sel_pair_id[idx_in_pair]]
                            self.tmp_right_dist_buf[cur_pair_id][idx_in_pair][n][:] = self.env.right_dist[sel_pair_id[idx_in_pair]]
                            self.tmp_success_buf[cur_pair_id][idx_in_pair][n][:] = 1 if infos['successes'][sel_pair_id[idx_in_pair]] != 0 else 0
                    elif self.config['task_name'] == 'kettle':
                        for idx_in_pair in range(2):
                            self.tmp_kettle_spout_pos_buf[cur_pair_id][idx_in_pair][n][:] = self.env.kettle_spout_pos[sel_pair_id[idx_in_pair]][:]
                            self.tmp_bucket_handle_pos_buf[cur_pair_id][idx_in_pair][n][:] = self.env.bucket_handle_pos[sel_pair_id[idx_in_pair]][:]
                            self.tmp_left_hand_ff_pos_buf[cur_pair_id][idx_in_pair][n][:] = self.env.left_hand_ff_pos[sel_pair_id[idx_in_pair]][:]
                            self.tmp_right_hand_ff_pos_buf[cur_pair_id][idx_in_pair][n][:] = self.env.right_hand_ff_pos[sel_pair_id[idx_in_pair]][:]
                            self.tmp_kettle_handle_pos_buf[cur_pair_id][idx_in_pair][n][:] = self.env.kettle_handle_pos[sel_pair_id[idx_in_pair]][:]
                            self.tmp_success_buf[cur_pair_id][idx_in_pair][n][:] = 1 if infos['successes'][sel_pair_id[idx_in_pair]] != 0 else 0

            else:
                for newk, sel_pair_id in enumerate(selected_pairs_ids):
                    cur_pair_id = self.cur_ep_start_id + newk
                    if self.config['task_name'] == 'hand_over':
                        for idx_in_pair in range(2):
                            self.tot_object_pos_buf[cur_pair_id][idx_in_pair][n][:] = self.env.object_pos[sel_pair_id[idx_in_pair]][:]
                            self.tot_object_linvel_buf[cur_pair_id][idx_in_pair][n][:] = self.env.object_linvel[sel_pair_id[idx_in_pair]][:]
                            self.tot_success_buf[cur_pair_id][idx_in_pair][n][:] = 1 if infos['successes'][sel_pair_id[idx_in_pair]] != 0 else 0
                            self.tot_dist_to_tip_buf[cur_pair_id][idx_in_pair][n][:] = self.env.dist_to_tip[sel_pair_id[idx_in_pair]][:]
                            self.tot_dist_to_another_tip_buf[cur_pair_id][idx_in_pair][n][:] = self.env.dist_to_another_tip[sel_pair_id[idx_in_pair]][:]
                    elif self.config['task_name'] == 'swing_cup':
                        for idx_in_pair in range(2):
                            self.tot_object_linvel_buf[cur_pair_id][idx_in_pair][n][:] = self.env.object_linvel[sel_pair_id[idx_in_pair]][:]
                            self.tot_object_rot_buf[cur_pair_id][idx_in_pair][n][:] = self.env.object_rot3[sel_pair_id[idx_in_pair]][:]
                            self.tot_left_dist_buf[cur_pair_id][idx_in_pair][n][:] = self.env.left_dist[sel_pair_id[idx_in_pair]]
                            self.tot_right_dist_buf[cur_pair_id][idx_in_pair][n][:] = self.env.right_dist[sel_pair_id[idx_in_pair]]
                            self.tot_success_buf[cur_pair_id][idx_in_pair][n][:] = 1 if infos['successes'][sel_pair_id[idx_in_pair]] != 0 else 0
                    elif self.config['task_name'] == 'kettle':
                        for idx_in_pair in range(2):
                            self.tot_kettle_spout_pos_buf[cur_pair_id][idx_in_pair][n][:] = self.env.kettle_spout_pos[sel_pair_id[idx_in_pair]][:]
                            self.tot_bucket_handle_pos_buf[cur_pair_id][idx_in_pair][n][:] = self.env.bucket_handle_pos[sel_pair_id[idx_in_pair]][:]
                            self.tot_left_hand_ff_pos_buf[cur_pair_id][idx_in_pair][n][:] = self.env.left_hand_ff_pos[sel_pair_id[idx_in_pair]][:]
                            self.tot_right_hand_ff_pos_buf[cur_pair_id][idx_in_pair][n][:] = self.env.right_hand_ff_pos[sel_pair_id[idx_in_pair]][:]
                            self.tot_kettle_handle_pos_buf[cur_pair_id][idx_in_pair][n][:] = self.env.kettle_handle_pos[sel_pair_id[idx_in_pair]][:]
                            self.tot_success_buf[cur_pair_id][idx_in_pair][n][:] = 1 if infos['successes'][sel_pair_id[idx_in_pair]] != 0 else 0
            # endregion

            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards) # [2048,1]
            
            if 'consecutive_successes' in infos:
                self.game_successes = infos['consecutive_successes']

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            if mean_step_pref_rew is not None:
                scaled_pref_rewards = self.pref_scale * pref_rewards.unsqueeze(1)
                self.experience_buffer.update_data('pref_rewards', n, scaled_pref_rewards) # [2048,1]
                self.cur_pref_rewards += scaled_pref_rewards
                self.pref_rewards.update(self.cur_pref_rewards[env_done_indices])
                self.cur_pref_rewards = self.cur_pref_rewards * not_dones.unsqueeze(1)

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        self.cur_ep_start_id += len(selected_pairs_ids)
        mean_ep_pref_rew = traj_pref_rew_buf.mean().item()

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_pref_rewards = self.experience_buffer.tensor_dict['pref_rewards']
        if self.pref_predictor is None:
            mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        else:
            mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards + mb_pref_rewards)

        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        # region [DIY]
        if epoch_num % self.buf_interval == 0 and epoch_num > 210:
            if epoch_num < 310:
                # print("!!! Making the initial training set.")
                assert cur_pair_id == self.num_tot_pairs - 1 # 599
                self.pref_train_data = None
                if self.config['task_name'] == 'hand_over':
                    self.pref_train_data = {
                        'object_pos_buf': self.tot_object_pos_buf,
                        'object_linvel_buf': self.tot_object_linvel_buf,
                        'success_buf': self.tot_success_buf,
                        'dist_buf': self.tot_dist_to_tip_buf,
                        'dist_another_buf': self.tot_dist_to_another_tip_buf
                    }
                elif self.config['task_name'] == 'swing_cup':
                    self.pref_train_data = {
                        'object_linvel_buf': self.tot_object_linvel_buf,
                        'object_rot_buf': self.tot_object_rot_buf,
                        'success_buf': self.tot_success_buf,
                        'left_dist_buf': self.tot_left_dist_buf,
                        'right_dist_buf': self.tot_right_dist_buf
                    }
                elif self.config['task_name'] == 'kettle':
                    self.pref_train_data = {
                        'kettle_spout_pos_buf': self.tot_kettle_spout_pos_buf,
                        'kettle_handle_pos_buf': self.tot_kettle_handle_pos_buf,
                        'bucket_handle_pos_buf': self.tot_bucket_handle_pos_buf,
                        'left_hand_ff_pos_buf': self.tot_left_hand_ff_pos_buf,
                        'right_hand_ff_pos_buf': self.tot_right_hand_ff_pos_buf,
                        'success_buf': self.tot_success_buf,
                    }
                # print("$$$ The initial data states are collected.")

                self.pref_train_data = self.gpt_gen_pref_labels(self.pref_train_data)
                # print("$$$ The initial data labels are generated.")

                os.makedirs(os.path.join(self.train_dir, 'traj_pairs'), exist_ok=True)
                torch.save(self.pref_train_data, os.path.join(self.train_dir, 'traj_pairs', 'traj_pairs_initial600.pt'))

                self.cur_ep_start_id = 0

            else:
                assert cur_pair_id == self.num_collection - 1 # 99
                if self.config['task_name'] == 'hand_over':
                    new_pref_train_data = {
                        'object_pos_buf': self.tmp_object_pos_buf,
                        'object_linvel_buf': self.tmp_object_linvel_buf,
                        'success_buf': self.tmp_success_buf,
                        'dist_buf': self.tmp_dist_to_tip_buf,
                        'dist_another_buf': self.tmp_dist_to_another_tip_buf
                    }
                elif self.config['task_name'] == 'swing_cup':
                    new_pref_train_data = {
                        'object_linvel_buf': self.tmp_object_linvel_buf,
                        'object_rot_buf': self.tmp_object_rot_buf,
                        'left_dist_buf': self.tmp_left_dist_buf,
                        'right_dist_buf': self.tmp_right_dist_buf,
                        'success_buf': self.tmp_success_buf,
                    }
                elif self.config['task_name'] == 'kettle':
                    new_pref_train_data = {
                        'kettle_spout_pos_buf': self.tmp_kettle_spout_pos_buf,
                        'kettle_handle_pos_buf': self.tmp_kettle_handle_pos_buf,
                        'bucket_handle_pos_buf': self.tmp_bucket_handle_pos_buf,
                        'left_hand_ff_pos_buf': self.tmp_left_hand_ff_pos_buf,
                        'right_hand_ff_pos_buf': self.tmp_right_hand_ff_pos_buf,
                        'success_buf': self.tmp_success_buf,
                    }
                mixed_pref_train_data = self.create_mixed_data(input_data=new_pref_train_data)

                mixed_pref_train_data = self.gpt_gen_pref_labels(mixed_pref_train_data)

                torch.save(mixed_pref_train_data, os.path.join(self.train_dir, 'traj_pairs', 'traj_pairs_{}.pt'.format(epoch_num)))
                self.pref_train_data = self.update_train_data_queue(new_data=mixed_pref_train_data)

                # empty the buffers
                if self.config['task_name'] == 'hand_over':
                    self.tmp_object_pos_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.object_pos[0]),
                                                        device=self.ppo_device, dtype=torch.float, requires_grad=False)
                    self.tmp_object_linvel_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.object_linvel[0]),
                                                        device=self.ppo_device, dtype=torch.float, requires_grad=False)
                    self.tmp_success_buf = torch.zeros(self.num_collection, 2, self.horizon_length, 1,
                                                        device=self.ppo_device, dtype=torch.float, requires_grad=False)
                    self.tmp_dist_to_tip_buf = torch.zeros(self.num_collection, 2, self.horizon_length, 5,
                                                        device=self.ppo_device, dtype=torch.float, requires_grad=False)
                    self.tmp_dist_to_another_tip_buf = torch.zeros(self.num_collection, 2, self.horizon_length, 5,
                                                        device=self.ppo_device, dtype=torch.float, requires_grad=False)
                elif self.config['task_name'] == 'swing_cup':
                    self.tmp_object_linvel_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.object_linvel[0]),
                                                        device=self.ppo_device, dtype=torch.float, requires_grad=False)
                    self.tmp_object_rot_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.object_rot3[0]),
                                                        device=self.ppo_device, dtype=torch.float, requires_grad=False)
                    self.tmp_success_buf = torch.zeros(self.num_collection, 2, self.horizon_length, 1,
                                                        device=self.ppo_device, dtype=torch.float, requires_grad=False)
                    self.tmp_left_dist_buf = torch.zeros(self.num_collection, 2, self.horizon_length, 1,
                                                        device=self.ppo_device, dtype=torch.float, requires_grad=False)
                    self.tmp_right_dist_buf = torch.zeros(self.num_collection, 2, self.horizon_length, 1,
                                                        device=self.ppo_device, dtype=torch.float, requires_grad=False)
                elif self.config['task_name'] == 'kettle':
                    self.tmp_kettle_spout_pos_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.kettle_spout_pos[0]),
                                                        device=self.ppo_device, dtype=torch.float, requires_grad=False)
                    self.tmp_bucket_handle_pos_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.bucket_handle_pos[0]),
                                                        device=self.ppo_device, dtype=torch.float, requires_grad=False)
                    self.tmp_left_hand_ff_pos_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.left_hand_ff_pos[0]),
                                                        device=self.ppo_device, dtype=torch.float, requires_grad=False)
                    self.tmp_right_hand_ff_pos_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.right_hand_ff_pos[0]),
                                                        device=self.ppo_device, dtype=torch.float, requires_grad=False)
                    self.tmp_success_buf = torch.zeros(self.num_collection, 2, self.horizon_length, 1,
                                                        device=self.ppo_device, dtype=torch.float, requires_grad=False)
                    self.tmp_kettle_handle_pos_buf = torch.zeros(self.num_collection, 2, self.horizon_length, len(self.env.kettle_handle_pos[0]),
                                                        device=self.ppo_device, dtype=torch.float, requires_grad=False)
                self.cur_ep_start_id = 0
        # endregion

        return batch_dict

    def play_steps_rnn(self):
        update_list = self.update_list
        mb_rnn_states = self.mb_rnn_states
        step_time = 0.0

        for n in range(self.horizon_length):
            if n % self.seq_len == 0:
                for s, mb_s in zip(self.rnn_states, mb_rnn_states):
                    mb_s[n // self.seq_len,:,:,:] = s

            if self.has_central_value:
                self.central_value_net.pre_step_rnn(n)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.rnn_states = res_dict['rnn_states']
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones.byte())

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)

            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            if len(all_done_indices) > 0:
                if self.zero_rnn_on_done:
                    for s in self.rnn_states:
                        s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0
                if self.has_central_value:
                    self.central_value_net.post_step_rnn(all_done_indices)

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()

        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values
        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        states = []
        for mb_s in mb_rnn_states:
            t_size = mb_s.size()[0] * mb_s.size()[2]
            h_size = mb_s.size()[3]
            states.append(mb_s.permute(1,2,0,3).reshape(-1,t_size, h_size))
        batch_dict['rnn_states'] = states
        batch_dict['step_time'] = step_time
        return batch_dict


class LAPPDiscreteA2CBase(LAPPA2CBase):

    def __init__(self, base_name, params):
        LAPPA2CBase.__init__(self, base_name, params)
    
        batch_size = self.num_agents * self.num_actors
        action_space = self.env_info['action_space']
        if type(action_space) is gym.spaces.Discrete:
            self.actions_shape = (self.horizon_length, batch_size)
            self.actions_num = action_space.n
            self.is_multi_discrete = False
        if type(action_space) is gym.spaces.Tuple:
            self.actions_shape = (self.horizon_length, batch_size, len(action_space)) 
            self.actions_num = [action.n for action in action_space]
            self.is_multi_discrete = True
        self.is_discrete = True

    def init_tensors(self):
        LAPPA2CBase.init_tensors(self)
        self.update_list = ['actions', 'neglogpacs', 'values']
        if self.use_action_masks:
            self.update_list += ['action_masks']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones']

    def train_epoch(self):
        super().train_epoch()

        self.set_eval()
        play_time_start = time.time()

        if self.is_rnn:
            batch_dict = self.play_steps_rnn()
        else:
            batch_dict = self.play_steps()

        self.set_train()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        a_losses = []
        c_losses = []
        entropies = []
        kls = []
        if self.has_central_value:
            self.train_central_value()

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.rank_size

            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)
            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul

    def prepare_dataset(self, batch_dict):
        rnn_masks = batch_dict.get('rnn_masks', None)
        returns = batch_dict['returns']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        dones = batch_dict['dones']
        rnn_states = batch_dict.get('rnn_states', None)
        
        obses = batch_dict['obses']
        advantages = returns - values

        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
            
        
        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks

        if self.use_action_masks:
            dataset_dict['action_masks'] = batch_dict['action_masks']

        self.dataset.update_values_dict(dataset_dict)
        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['dones'] = dones
            dataset_dict['obs'] = batch_dict['states'] 
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def train(self):
        self.init_tensors()
        self.mean_rewards = self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        # self.frame = 0  # loading from checkpoint
        self.obs = self.env_reset()

        if self.multi_gpu:
            torch.cuda.set_device(self.rank)
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            total_time += sum_time
            curr_frames = self.curr_frames * self.rank_size if self.multi_gpu else self.curr_frames
            self.frame += curr_frames
            should_exit = False

            if self.rank == 0:
                self.diagnostics.epoch(self, current_epoch = epoch_num)
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time

                frame = self.frame // self.num_agents

                print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time, 
                                epoch_num, self.max_epochs, frame, self.max_frames)

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                                a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, 
                                scaled_time, scaled_play_time, curr_frames)

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)
                        self.writer.add_scalar('shaped_' + rewards_name + '/step'.format(i), mean_shaped_rewards[i], frame)
                        self.writer.add_scalar('shaped_' + rewards_name + '/iter'.format(i), mean_shaped_rewards[i], epoch_num)
                        self.writer.add_scalar('shaped_' + rewards_name + '/time'.format(i), mean_shaped_rewards[i], total_time)


                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    # removed equal signs (i.e. "rew=") from the checkpoint name since it messes with hydra CLI parsing
                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards <= self.last_mean_rewards):
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Maximum reward achieved. Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True
                    
                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.bool().item()

            if should_exit:
                return self.last_mean_rewards, epoch_num


class LAPPContinuousA2CBase(LAPPA2CBase):

    def __init__(self, base_name, params):
        LAPPA2CBase.__init__(self, base_name, params)

        self.is_discrete = False
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]
        self.bounds_loss_coef = self.config.get('bounds_loss_coef', None)

        self.clip_actions = self.config.get('clip_actions', True)

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)

    def preprocess_actions(self, actions):
        if self.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = rescale_actions(self.actions_low, self.actions_high, clamped_actions)
        else:
            rescaled_actions = actions

        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()

        return rescaled_actions

    def init_tensors(self):
        LAPPA2CBase.init_tensors(self)
        self.update_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones']

    def train_epoch(self, epoch_num):
        super().train_epoch()

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps(epoch_num)

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()
        if self.has_central_value:
            self.train_central_value()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)
                if self.schedule_type == 'legacy':
                    av_kls = kl
                    if self.multi_gpu:
                        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                        av_kls /= self.rank_size
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.rank_size
            if self.schedule_type == 'standard':
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        advantages = returns - values

        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['dones'] = dones
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch(epoch_num)
            total_time += sum_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False
            
            if self.rank == 0:
                self.diagnostics.epoch(self, current_epoch = epoch_num)
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames * self.rank_size if self.multi_gpu else self.curr_frames
                self.frame += curr_frames

                print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time, 
                                epoch_num, self.max_epochs, frame, self.max_frames)

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                                a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame,
                                scaled_time, scaled_play_time, curr_frames)

                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

                if self.has_soft_aug:
                    self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    mean_successes = None
                    try:
                        mean_successes = self.game_successes.mean().cpu().numpy()
                        print(f"@@@ Current success is {mean_successes:.3f};", end=" ")
                    except:
                        pass 
                    self.mean_rewards = mean_rewards[0]
                    
                    print(f"Current reward is {self.mean_rewards:.3f};", end=' ')
                    if self.pref_predictor is None:
                        print("No pref reward")
                    else:
                        mean_pref_rewards = self.pref_rewards.get_mean()
                        print(f"Current pref reward is {mean_pref_rewards[0]:.3f}")

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)
                        self.writer.add_scalar('shaped_' + rewards_name + '/step'.format(i), mean_shaped_rewards[i], frame)
                        self.writer.add_scalar('shaped_' + rewards_name + '/iter'.format(i), mean_shaped_rewards[i], epoch_num)
                        self.writer.add_scalar('shaped_' + rewards_name + '/time'.format(i), mean_shaped_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    # if self.save_freq > 0:
                    #     if (epoch_num % self.save_freq == 0) and (mean_rewards[0] <= self.last_mean_rewards):
                    #         self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_successes is not None:
                        if mean_successes > self.last_mean_successes and epoch_num >= self.save_best_after:
                            print('saving next best successes: ', mean_successes)
                            self.last_mean_successes = mean_successes
                            self.save(os.path.join(self.nn_dir, self.config['name']))
                            # self.save(os.path.join(self.nn_dir, 'successes_' + self.config['name'] + '_ep_' + str(epoch_num) \
                            # + '_rew_' + str(mean_successes).replace('[', '_').replace(']', '_')))
       
                    # if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                    #     print('saving next best rewards: ', mean_rewards)
                    #     self.last_mean_rewards = mean_rewards[0]
                        # self.save(os.path.join(self.nn_dir, self.config['name']))
                        # self.save(os.path.join(self.nn_dir, 'reward_' + self.config['name'] + '_ep_' + str(epoch_num) \
                        # + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))

                        # if 'score_to_win' in self.config:
                        #     if self.last_mean_rewards > self.config['score_to_win']:
                        #         print('Maximum reward achieved. Network won!')
                        #         self.save(os.path.join(self.nn_dir, checkpoint_name))
                        #         should_exit = True

                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf
                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num)))
                    # self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
                    #     + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0
            
            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num
