import os
import torch
import torch.nn.functional as F

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image
from torchvision import transforms

from src.common.utils import make_grid
from src.experiment.experiment import Experiment
from src.datasets.diffusion_synthesis import get_unwrapped_sprite_dataset

def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size = config.eval_batch_size, 
        generator=torch.manual_seed(config.seed),
        num_inference_steps=config.__dict__.get('num_inference_timesteps', config.num_train_timesteps)
    ).images

    image_grid = make_grid(images, rows=4, cols=4)

    test_dir = os.path.join(config.project_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


class BaselineImage(Experiment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dataset = get_unwrapped_sprite_dataset(self.config.data_dir)

        preprocess = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGBA")),
                transforms.ToTensor(),
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.RandomRotation(15),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ElasticTransform(alpha=30., sigma=8., interpolation=transforms.InterpolationMode.NEAREST),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def transform(examples):
            images = [preprocess(image.convert("RGB")) for image in examples["image"]]
            return {"images": images}

        self.dataset.set_transform(transform)

        self.train_dataloader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.config.train_batch_size, 
            shuffle=self.config.shuffle
        )

        # Set up model, optimizer, scheduler, etc.
        self.model = UNet2DModel(
            sample_size=self.config.image_size,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            layers_per_block=self.config.layers_per_block,
            block_out_channels=[block[1] for block in self.config.down_blocks],
            down_block_types=[block[0] for block in self.config.down_blocks],
            up_block_types=self.config.up_blocks
            dropout=self.config.dropout
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.config.num_train_timesteps,
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=(len(self.train_dataloader) * self.config.num_epochs)
        )

        self.args = (
            self.config,
            self.model,
            self.noise_scheduler,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler
        )

    def train_setup(self):
        super().train_setup()

        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

    def train_single_iteration(self, idx, batch, *args, **kwargs):
        clean_images = batch['images']
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
        ).long()

        noisy_images = self.noise_scheduler.add_noise(
            clean_images, noise, timesteps
        )

        with self.accelerator.accumulate(self.model):
            noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            self.accelerator.backward(loss)

            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
    
    def train_evaluation(self, epoch, *args, **kwargs):
        pipeline = DDPMPipeline(
            unet=self.accelerator.unwrap_model(self.model),
            scheduler=self.noise_scheduler,
        )

        if (epoch + 1) % self.config.save_image_epochs == 0 or epoch == self.config.num_epochs - 1:
            evaluate(self.config, epoch, pipeline)

        if (epoch + 1) % self.config.save_model_epochs == 0 or epoch == self.config.num_epochs - 1:
            pipeline.save_pretrained(self.config.project_dir)

    
if __name__ == "__main__":
    training_metadata = {
        'experiment_name': 'baseline_img',
        'project_dir': '_testing_runs',
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
                ('AttnDownBlock2D', 64),
                ('DownBlock2D', 128),
                ('DownBlock2D', 128),
                ('DownBlock2D', 256),
                ('DownBlock2D', 256),
        ],
        'up_blocks': [
            'UpBlock2D',
            'UpBlock2D',
            'UpBlock2D',
            'UpBlock2D',
            'AttnUpBlock2D',
            'UpBlock2D',
        ]
    }

    training_hyperparameters = {
        'train_batch_size': 64,
        'eval_batch_size': 16,
        'gradient_accumulation_steps': 1,
        'learning_rate': 1e-4,
        'lr_warmup_steps': 200,
        'dropout': 0.3,
        'save_image_epochs': 1,
        'save_model_epochs': 1,
        'seed': 42,
        'num_train_timesteps': 200, 
        'num_batches': None,
        'num_epochs': 1500,
        'shuffle': True
    }

    training_config = {**training_metadata, 
                       **model_parametrs, 
                       **training_hyperparameters}

    experiment = BaselineImage(**training_config)
    experiment.run(num_processes=1)