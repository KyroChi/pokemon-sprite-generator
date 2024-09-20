import numpy as np
import os
import random
import torch
import torch.nn.functional as F

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from src.common.utils import make_grid
from src.experiment.experiment import Experiment
from src.datasets.diffusion_synthesis import get_unwrapped_sprite_dataset, get_full_art_conditional_dataset

def feature_mapping(x, B):
    # B = (num_features, in_channels)
    # x = (B, in_channels, H, W)
    # x_proj = (B, 2 * num_features, H, W)
    x_proj = torch.einsum("fc, Bchw -> Bfhw", B, (2 * np.pi * x))
    x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)
    return x_proj

class RandomCenteredCrop(transforms.RandomCrop):
    """Randomly crop the input image, but instead of sampling from a uniform distribution
    to determine the crop center, this transform samples from a normal distribution centered 
    at the center of the image.
    """
    def __init__(self, size, mu=0.5, sigma=0.1):
        super().__init__(size)
        self.mu = mu
        self.sigma = sigma

    def get_params(self, img, output_size):
        """Get parameters for `crop` for a random crop centered around the image center."""
        _, h, w = TF.get_dimensions(img)
        th, tw = output_size

        # Sample from a normal distribution centered at the image center
        center_x = int(random.gauss(self.mu, self.sigma) * w)
        center_y = int(random.gauss(self.mu, self.sigma) * h)

        # Ensure the crop box is within the image bounds
        x1 = max(0, min(center_x - tw // 2, w - tw))
        y1 = max(0, min(center_y - th // 2, h - th))

        return y1, x1, th, tw

@torch.no_grad()
def sample_iadb(model, x0, nb_step, feature_mapping):
    model.eval()
    x_alpha = x0
    for t in range(nb_step):
        alpha_start = (t/nb_step)
        alpha_end =((t+1)/nb_step)

        fourier = feature_mapping(x_alpha)
        network_in = torch.cat([x_alpha, fourier], dim=1)

        d = model(network_in, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        x_alpha = x_alpha + (alpha_end-alpha_start)*d

    model.train()
    return x_alpha

def evaluate(config, model, epoch, x0_eval, feature_mapping):
    steps = config.num_inference_timesteps

    sample = (sample_iadb(model, x0_eval, steps, feature_mapping) * 0.5) + 0.5
    dirname = os.path.join(config.project_dir, 'output')
    os.makedirs(dirname, exist_ok=True)
    save_image(sample, f'{dirname}/export_{str(epoch).zfill(8)}.png')

class BaselineImageIADB(Experiment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.dataset = get_unwrapped_sprite_dataset(self.config.data_dir, samples_per_pokemon=30)
        self.dataset = get_full_art_conditional_dataset(self.config.data_dir)

        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.config.image_size, self.config.image_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomRotation(180, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomHorizontalFlip(), # This is regularizing
                transforms.RandomVerticalFlip(), # This is regularizing
                transforms.Resize((48, 48), interpolation=transforms.InterpolationMode.NEAREST),
                # RandomCenteredCrop((64, 64), mu=0.5, sigma=0.12),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def transform(examples):
            images = [preprocess(image.convert("RGBA")) for image in examples["image"]]
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
            in_channels=self.config.in_channels + 2 * self.config.ff_num_features,
            out_channels=self.config.out_channels,
            layers_per_block=self.config.layers_per_block,
            block_out_channels=[block[1] for block in self.config.down_blocks],
            down_block_types=[block[0] for block in self.config.down_blocks],
            up_block_types=self.config.up_blocks,
            dropout=self.config.dropout,
        )

        self.model.load_state_dict(
            load_file('/home/kyle/projects/pokemon_diffusion/_testing_runs/baseline_img_iadb_ff/_good_run_overnight/models/model_373.pt/model.safetensors')
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
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler
        )

        self.x0_eval = None

    def train_setup(self):
        super().train_setup()

        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        B = self.config.ff_scale * torch.randn(self.config.ff_num_features, self.config.in_channels)
        B = B.to(self.accelerator.device)

        self.feature_mapping = lambda x: feature_mapping(x, B)

    def train_single_iteration(self, idx, batch, *args, **kwargs):
        x1 = batch["images"] # Images are assumed to already be normalized
        x0 = torch.randn_like(x1).to(x1.device)

        if self.x0_eval is None:
            self.x0_eval = x0

        bs = x0.shape[0]

        alpha = torch.rand(bs, device=x1.device)
        x_alpha = torch.lerp(x0, x1, alpha.view(-1, 1, 1, 1))

        fourier = self.feature_mapping(x_alpha)
        x_alpha = torch.cat([x_alpha, fourier], dim=1)

        with self.accelerator.accumulate(self.model):
            diff_pred = self.model(x_alpha, alpha)['sample']
            loss = F.mse_loss(diff_pred, x1 - x0)
            self.accelerator.backward(loss)

            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return {
            "loss": loss.detach().item(),
            "lr": self.lr_scheduler.get_last_lr()[0]
        }
    
    def train_evaluation(self, epoch, *args, **kwargs):
        if (epoch + 1) % self.config.save_image_epochs == 0 or epoch == self.config.num_epochs - 1:
            evaluate(self.config, self.model, epoch + 1, self.x0_eval, self.feature_mapping)

        if (epoch + 1) % self.config.save_model_epochs == 0 or epoch == self.config.num_epochs - 1:
            self.accelerator.save_model(self.model, os.path.join(self.config.project_dir, "models", f"model_{epoch + 1}.pt"))

if __name__ == "__main__":
    training_metadata = {
        'experiment_name': 'baseline_img_iadb_ff',
        'project_dir': '/home/kyle/projects/pokemon_diffusion/_testing_runs',
        'mixed_precision': 'fp16',
        'log_with': 'tensorboard',
        'data_dir': '/home/kyle/projects/pokemon_data/data',
        # 'data_dir': '/home/kyle/learning_diffusion/tools/pokemon_dataset/data/scraper_test/gen_v_unwrapped_sprites'
    }

    model_parametrs = {
        'image_size': 250,
        'in_channels': 4,
        'out_channels': 4,
        'layers_per_block': 3,
        'down_blocks': [
                ('DownBlock2D', 64),
                ('DownBlock2D', 64),
                ('AttnDownBlock2D', 128),
                ('DownBlock2D', 128),
                ('AttnDownBlock2D', 256),
                ('DownBlock2D', 256),
        ],
        'up_blocks': [
            'UpBlock2D',
            'AttnUpBlock2D',
            'UpBlock2D',
            'AttnUpBlock2D',
            'UpBlock2D',
            'UpBlock2D',
        ]
    }

    training_hyperparameters = {
        'num_inference_timesteps': 50,
        'train_batch_size': 64,
        'eval_batch_size': 16,
        'gradient_accumulation_steps': 1,
        'learning_rate': 1e-4 / 20,
        'lr_warmup_steps': 200,
        'save_image_epochs': 1,
        'save_model_epochs': 1,
        'dropout': 0.3,
        'seed': 42,
        'num_train_timesteps': 200, 
        'num_batches': None,
        'num_epochs': 1500,
        'shuffle': True,
        'ff_num_features': 16,
        'ff_scale': 10.,
    }

    training_config = {**training_metadata, 
                       **model_parametrs, 
                       **training_hyperparameters}

    experiment = BaselineImageIADB(**training_config)
    experiment.run(num_processes=1)