from baseline_img_iadb import BaselineImageIADB

if __name__ == "__main__":
    training_metadata = {
        'experiment_name': 'baseline_img_iadb_overfit',
        'project_dir': '/home/kyle/projects/pokemon_diffusion/_testing_runs',
        'mixed_precision': 'fp16',
        'log_with': 'tensorboard',
        'data_dir': '/home/kyle/learning_diffusion/tools/pokemon_dataset/data/scraper_test/gen_v_unwrapped_sprites'
    }

    model_parametrs = {
        'image_size': 64,
        'in_channels': 3,
        'out_channels': 3,
        'layers_per_block': 3,
        'down_blocks': [
                ('DownBlock2D', 64),
                ('DownBlock2D', 64),
                ('DownBlock2D', 128),
                ('DownBlock2D', 128),
                ('AttnDownBlock2D', 256),
                ('DownBlock2D', 256),
        ],
        'up_blocks': [
            'UpBlock2D',
            'AttnUpBlock2D',
            'UpBlock2D',
            'UpBlock2D',
            'UpBlock2D',
            'UpBlock2D',
        ]
    }

    training_hyperparameters = {
        'num_inference_timesteps': 50,
        'train_batch_size': 64,
        'eval_batch_size': 16,
        'gradient_accumulation_steps': 1,
        'learning_rate': 1e-4,
        'lr_warmup_steps': 200,
        'save_image_epochs': 25,
        'save_model_epochs': 25,
        'seed': 42,
        'num_train_timesteps': 200, 
        'num_batches': 1,
        'num_epochs': 1500,
        'shuffle': True
    }

    training_config = {**training_metadata, 
                       **model_parametrs, 
                       **training_hyperparameters}

    experiment = BaselineImageIADB(**training_config)
    experiment.run(num_processes=1)