import os
import time

from launch_utils import (create_log_dirs, get_argument_parser,
                          get_experience_streams, set_seeds)
from Source import architecture, learner

if __name__ == "__main__":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    args = get_argument_parser()
    args.use_wandb = True  # Ensure W&B is enabled
    
    current_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # print(f'current_TIME:{current_time}')
    print(f'args:{args}')

    log_dirpath = create_log_dirs(args)
    print(f'log_dirpath:{log_dirpath}')
    
    with open(os.path.join(log_dirpath, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
    
    set_seeds(args.seed)
    scenario, input_size, output_size, task2classes = get_experience_streams(args)
    backbone = architecture.get_backbone(args, input_size, output_size)
    nice = learner.Learner(args, backbone, scenario, input_size, task2classes, log_dirpath)

    start = time.time()
    nice.learn_all_episodes()
    print(f'time:{abs(start-time.time())}')

    # Save the model after training
    nice.save_model()