from baseline_img import BaselineImage

if __name__ == "__main__":
    training_metadata = {
        'experiment_name': 'baseline_img_overfit',
        'project_dir': '_testing_runs',
        'mixed_precision': 'fp16',
        'log_with': 'tensorboard',
        'data_dir': '/home/kyle/learning_diffusion/tools/pokemon_dataset/data/scraper_test/gen_v_unwrapped_sprites'
    }

    model_parametrs = {
        'image_size': 64,
        'in_channels': 3,
        'out_channels': 3,
        'layers_per_block': 2,
        'down_blocks': [
                ('DownBlock2D', 128),
                ('DownBlock2D', 128),
                ('DownBlock2D', 256),
                ('DownBlock2D', 256),
                ('AttnDownBlock2D', 512),
                ('DownBlock2D', 512),
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
        'train_batch_size': 64,
        'eval_batch_size': 16,
        'gradient_accumulation_steps': 1,
        'learning_rate': 1e-4,
        'lr_warmup_steps': 200,
        'save_image_epochs': 10,
        'save_model_epochs': 10,
        'seed': 42,
        'num_train_timesteps': 200,
        'num_batches': 1,
        'num_epochs': 100,
        'shuffle': False
    }

    training_config = {**training_metadata, 
                       **model_parametrs, 
                       **training_hyperparameters}

    experiment = BaselineImage(**training_config)
    experiment.run(num_processes=1)