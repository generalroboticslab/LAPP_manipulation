# LAPP: Large Language Model Feedback for Preference-Driven Reinforcement Learning

[Pingcheng Jian](https://pingcheng-jian.github.io/),
[Xiao Wei](https://www.linkedin.com/in/xiao-wei-36a10b325/),
[Yanbaihui Liu](https://www.linkedin.com/in/yanbaihui-liu-19077216b/),
[Samuel A. Moore](https://samavmoore.github.io/),
[Michael M. Zavlanos](https://mems.duke.edu/faculty/michael-zavlanos),
[Boyuan Chen](http://boyuanchen.com/)
<br>
Duke University
<br>

## Environment Setup

> [OS Version]
> The tested OS is Ubuntu 22.04.5 LTS

1. Create a new conda environment with:
```
conda create -n LAPP_man python=3.8
conda activate LAPP_man
```

2. Install IsaacGym (tested with `Preview Release 4/4`). Follow the [instruction](https://developer.nvidia.com/isaac-gym) to download the package.
```
tar -xvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e .
```
To test installation, you can run:
```
python examples/joint_monkey.py
```

3. Install *LAPP_manipulation*
```
git clone xxx
cd LAPP_manipulation; pip install -e .
cd isaacgymenvs; pip install -e .
cd ../rl_games; pip install -e .
```

4. Update *numpy* version to avoid attribute error
```
pip install numpy==1.21.0
```

5. LAPP (Manipulation) currently uses OpenAI API for language model queries. The way to set up is adding a *txt* file containing your own API key in the file `api_key/personal_api_key.txt`.

6. There is chance you encounter some path error like *"libpython3.8.so.1.0: cannot open shared object file: No such file or directory"*. To solve this, the following command works for the author:
```
export LD_LIBRARY_PATH=$(python3.8 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
```

## Getting Started
### Running our experiments
1. We have written a running script in main directory, `LAPP_demo.sh`. Before running the script, remember to
- Put your api key in the file `api_key/personal_api_key.txt`
- And enable the running of the script
```
chmod +x LAPP_demo.sh
```

2. The arguments are briefly described as:
- **task_name**: the task to run LAPP on, the provided code contains *hand_over*, *swing_cup* and *kettle* tasks
- **max_iter**: the total iterations of learning. As reference, 2000 can be enough for most training process to converge
- **pref_scale**: the weight of *preference rewards* added to the basic reward calculated with reward function
- **key_path**: specifies which file to read api key. Usually using default setting is enough
- **device**: the device you want to run your experiment on. Using number to indicate
- **test**: if you want to deploy our given checkpoints (to be discusses below)

2. Under `isaacgymenvs/isaacgymenvs/LAPP_results/` you can find the results of your run. If there're too many recordings and you are lost, you can refer to the training log, where you can find the saving information like *=> saving checkpoint 'LAPP_results/ShadowHandOverGPT/{time}/nn/last_ShadowHandOverGPT_ep_450.pth'*
With this information, you should be able to find the directory of your results, including
- **checkpoints**: of the best and last policy model
- **environment code**: that is specific to the task
- **configuration file**: of the whole experiment

### Deploy our checkpoints
In the repository, we have also enclosed the chosen checkpoints (for task *hand_over*, *swing_cup* and *kettle*) to review the work we have done. To deploy our checkpoints, you can run the following command:
```
cd isaacgymenvs/isaacgymenvs
python LAPP_main.py --task_name {} --test
```
And you can find the rendered video named like `{task_name}_render.mp4` under the same directory.

Besides, the running logs of experiments picked for figure plotting are provided, under directory `LAPP_ckpt/`.

## Further Customization
If the user want to feed their customized environment information for the LLM, following modifications should be made:
- Add your own prompt template under directory `LAPP_prompt_library/`
- In `isaacgymenvs/isaacgymenvs/LAPP_main.py`, you should modify the *task_name* list, and also specify the path of reading the corresponding config files
- In the corresponding task file, i.e. `isaacgymenvs/isaacgymenvs/tasks/shadow_hand_pen.py`, please design the information you would like to collect; also, you can design your *own reward function*
	- **Important note**: you need a function *LAPP_init_buf()* to claim the variable before reference to avoid error, which you can refer to other files for example
- In `rl_games/rl_games/common/LAPP_a2c_common.py`, there're several regions/comments labeled as *\[DIY\]*, for these parts, you can modify to your need
- In `rl_games/rl_games/common/LAPP_transformer/transformer_trainer.py`, there is one region/comment labeled as *\[DIY\]*, where you can modify the information you pass in for transformer training

## Some tips
1. The coefficient before *apply_force* in some task code is quite large, causing the possibility of simulation "explosion". In other words, some *NaN/Inf* will appear in data because the force is abnormally large, which causes the training and simulation to stop. For a more stable training, you can decrease the coefficient.
2. The training results can vary greatly, due to the randomness nature of LLM queries. So it may happen that you can not reach same high success rate as our result, but you should witness some good results if you repeat for some times.

## Locomotion
For the code of the locomotion tasks, check out: https://github.com/generalroboticslab/LAPP

## Acknowledgement

This project refers to the github repositories [Unitree RL GYM](https://github.com/unitreerobotics/unitree_rl_gym), 
[RSL RL](https://github.com/leggedrobotics/rsl_rl), and 
[Isaac Gym](https://github.com/isaac-sim/IsaacGymEnvs).

## License
This repository is released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details.
