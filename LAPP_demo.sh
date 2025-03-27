cd "$(dirname "$0")/isaacgymenvs/isaacgymenvs"
nohup python -u LAPP_main.py \
    --task_name kettle \
    --max_iter 450 \
    --pref_scale 0.01 \
    --key_path personal \
    --device 6 \
    > kettle_demo_log.out \
    2>&1 &