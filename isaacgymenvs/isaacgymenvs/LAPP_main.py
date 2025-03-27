import logging
import os
import datetime

import isaacgym

# import hydra
from hydra.utils import to_absolute_path
from isaacgymenvs.tasks import isaacgym_task_map
from omegaconf import DictConfig, OmegaConf
import gym
import sys 
import shutil
from pathlib import Path

import argparse

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

# ROOT_DIR = os.getcwd()
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict['params']['config']
    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict

def launch_rlg(cfg: DictConfig):
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous, amp_players, amp_models, amp_network_builder
    import isaacgymenvs

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    rank = int(os.getenv("LOCAL_RANK", "0"))
    cfg.seed += rank
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg.train.params.config.multi_gpu = cfg.multi_gpu

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            if cfg.test:
                envs = gym.wrappers.RecordVideo(
                    envs,
                    f"videos/{run_name}",
                    step_trigger=lambda step: (step % cfg.capture_video_freq == 0),
                    video_length=cfg.capture_video_len,
                )
            else:
                envs = gym.wrappers.RecordVideo(
                    envs,
                    f"videos/{run_name}",
                    step_trigger=lambda step: (step % cfg.capture_video_freq == 0) and (step > 0),
                    video_length=cfg.capture_video_len,
                )
        return envs

    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
    })
    
    # Save the environment code!

    exp_date = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    experiment_dir = os.path.join(f'LAPP_results/{cfg.train.params.config.name}', exp_date)
    os.makedirs(experiment_dir, exist_ok=True)

    try:
        output_file = f"{ROOT_DIR}/tasks/{cfg.task.env.env_name.lower()}.py"
        shutil.copy(output_file, f"{experiment_dir}/env.py")
    except:
        import re
        def camel_to_snake(name):
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        output_file = f"{ROOT_DIR}/tasks/{camel_to_snake(cfg.task.name)}.py"

        shutil.copy(output_file, f"{experiment_dir}/env.py")

    vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

    observer = RLGPUAlgoObserver()

    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    rlg_config_dict['params']['config']['log_dir'] = exp_date
    rlg_config_dict['params']['config']['train_dir'] = experiment_dir

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(observer)
    runner.load(rlg_config_dict)
    runner.reset()

    statistics = runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint' : cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    })

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # region [DIY]
    parser.add_argument('--task_name', type=str, choices=['hand_over', 
                                                          'swing_cup', 
                                                          'kettle'], default='hand_over')
    # endregion
    parser.add_argument('--max_iter', type=int, default=2500)
    parser.add_argument('--pref_scale', type=float, default=0.0)
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--key_path', type=str, choices=['personal', 'empty'], default='empty')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    # region [DIY]
    if args.task_name == 'hand_over':
        cfg = OmegaConf.load('./cfg/config.yaml')
        cfg.task = OmegaConf.load('./cfg/task/ShadowHandOverGPT.yaml')
        cfg.train = OmegaConf.load('./cfg/train/ShadowHandOverGPTPPO.yaml')
        cfg.task_name = 'ShadowHandOverGPT'  
    elif args.task_name == 'swing_cup':
        cfg = OmegaConf.load('./cfg/config.yaml')
        cfg.task = OmegaConf.load('./cfg/task/ShadowHandSwingCupGPT.yaml')
        cfg.train = OmegaConf.load('./cfg/train/ShadowHandSwingCupGPTPPO.yaml')
        cfg.task_name = 'ShadowHandSwingCupGPT'
    elif args.task_name == 'kettle':
        cfg = OmegaConf.load('./cfg/config.yaml')
        cfg.task = OmegaConf.load('./cfg/task/ShadowHandKettleGPT.yaml')
        cfg.train = OmegaConf.load('./cfg/train/ShadowHandKettleGPTPPO.yaml')
        cfg.task_name = 'ShadowHandKettleGPT'
    # endregion
        
    cfg.max_iterations = args.max_iter
    cfg.train.params.config.pref_scale = args.pref_scale
    cfg.train.params.config.key_path = args.key_path
    cfg.train.params.config.horizon_length = args.horizon
    cfg.sim_device = f'cuda:{args.device}'
    cfg.rl_device = cfg.sim_device
    cfg.train.params.config.device = cfg.sim_device
    cfg.train.params.config.task_name = args.task_name

    if args.pref_scale == 0.0:
        cfg.train.params.algo.name = 'a2c_continuous' # if not using LAPP, use original training schema
    else:
        cfg.train.params.algo.name = 'LAPP_a2c_continuous' # this is preset in training config

    cfg.test = args.test
    if cfg.test:
        cfg.headless = True
        cfg.force_render = False
        cfg.task.env.record = True
        cfg.task.enableCameraSensors = True
        cfg.train.params.algo.name = 'a2c_continuous'

    # print the current config dictionary for debug
    # ====================
    # cfg_dict = omegaconf_to_dict(cfg)
    # print_dict(cfg_dict)
    # ====================

    launch_rlg(cfg)
