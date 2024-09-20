"""
    Conditional baseline image generation model.

    We use the following class labels:
    1. is_legendary
    2. is_mythical
    3. color
    4. shape
    5. type 1
    6. type 2
    7. is_shiny
"""

import os
import random
import torch
import torch.nn.functional as F

from datasets import Dataset
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from src.common.utils import make_grid
from src.experiment.experiment import Experiment
from src.datasets.diffusion_synthesis import get_unwrapped_conditional_dataset

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
def cfg_sample_iadb(model, x0, cond, nb_steps, w):
    model.eval()
    x_alpha = x0

    cond = cond.to(x0.device)
    zero = torch.zeros_like(cond).to(x0.device)

    for t in range(nb_steps):
        alpha_start = (t/nb_steps)
        alpha_end =((t+1)/nb_steps)

        net_input_cond = torch.cat([x_alpha, cond], dim=1)
        net_input_uncond = torch.cat([x_alpha, zero], dim=1)

        d_cond = model(net_input_cond, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        d_uncond = model(net_input_uncond, torch.tensor(alpha_start, device=x_alpha.device))['sample']

        d = (1 + w) * d_cond - w * d_uncond

        x_alpha = x_alpha + (alpha_end - alpha_start) * d

    model.train()
    return x_alpha


def evaluate(config, model, epoch, x0_eval, cond_eval, n_interpolants=4):
    steps = config.num_inference_timesteps

    for ii in range(n_interpolants):
        weight = ii

        sample = (cfg_sample_iadb(
            model=model, 
            x0=x0_eval, 
            cond=cond_eval, 
            nb_steps=steps, 
            w=weight) * 0.5) + 0.5
        # image_grid = make_grid(sample, rows=4, cols=4)

        test_dir = os.path.join(config.project_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        save_image(sample, f"{test_dir}/{epoch:04d}_{ii}.png")


class BaselineImageConditionalIADB(Experiment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dataset = get_unwrapped_conditional_dataset(
            path=self.config.data_dir,
            get_shiny=True,
            samples_per_pokemon=30
        )

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.RandomHorizontalFlip(), # This is regularizing
            transforms.RandomVerticalFlip(), # This is regularizing
            transforms.RandomRotation(180, interpolation=transforms.InterpolationMode.NEAREST),
            # transforms.Resize((48, 48), interpolation=transforms.InterpolationMode.NEAREST),
            # RandomCenteredCrop((32, 32), mu=0.5, sigma=0.12),
            transforms.Normalize([0.5], [0.5]),   
        ])

        # preprocess = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((self.config.image_size, self.config.image_size)),
        #     # transforms.RandomRotation(35, interpolation=transforms.InterpolationMode.NEAREST),
        #     # transforms.RandomHorizontalFlip(), # This is regularizing
        #     # RandomCenteredCrop((64, 64), mu=0.5, sigma=0.12),
        #     transforms.Normalize([0.5], [0.5]),   
        # ])

        def transform(examples):
            images = [preprocess(image.convert("RGBA")) for image in examples["image"]]
            return {"images": images, "conditions": torch.Tensor(examples["conditions"])}
        
        self.dataset.set_transform(transform)

        self.train_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.train_batch_size,
            shuffle=self.config.shuffle
        )

        self.model = UNet2DModel(
            sample_size=self.config.image_size,
            in_channels=self.config.in_channels + 7,
            out_channels=self.config.out_channels,
            layers_per_block=self.config.layers_per_block,
            block_out_channels=[block[1] for block in self.config.down_blocks],
            down_block_types=[block[0] for block in self.config.down_blocks],
            up_block_types=self.config.up_blocks,
            dropout=self.config.dropout
        )

        self.model.load_state_dict(
            load_file('/home/kyle/projects/pokemon_diffusion/_testing_runs/baseline_img_conditional_iadb/2024-09-06_11-29-44/models/model_77.pt/model.safetensors')
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

        torch.save(self.config, os.path.join(self.config.project_dir, "config.pt"))

        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = \
        self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

    def train_single_iteration(self, idx, batch, *args, **kwargs):
        x1 = batch["images"]
        x0 = torch.randn_like(x1).to(x1.device)
        cond = batch["conditions"]

        bs, _, h, w = x1.shape
        cond = cond.to(x1.device)
        cond = cond.view(bs, -1).unsqueeze(-1).unsqueeze(-1).expand(bs, -1, h, w)

        if self.x0_eval is None:
            self.x0_eval = x0
            self.cond_eval = cond

        # Randomly mask out elements of the condition vector
        # Following Ho, Salimans, et al. (2020), we train the unconditional model
        # 10% of the time.
        if random.random() < 0.1:
            cond = torch.zeros_like(cond)

        alpha = torch.rand(bs, device=x1.device)
        x_alpha = torch.lerp(x0, x1, alpha.view(-1, 1, 1, 1))

        net_input = torch.cat([x_alpha, cond], dim=1)

        with self.accelerator.accumulate(self.model):
            diff_pred = self.model(
                sample=net_input, 
                timestep=alpha)["sample"]
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
            evaluate(self.config, self.model, epoch, self.x0_eval, cond_eval=self.cond_eval)

        if (epoch + 1) % self.config.save_model_epochs == 0 or epoch == self.config.num_epochs - 1:
            self.accelerator.save_model(self.model, os.path.join(self.config.project_dir, "models", f"model_{epoch + 1}.pt"))


if __name__ == "__main__":
    training_metadata = {
        'experiment_name': 'baseline_img_conditional_iadb',
        'project_dir': '/home/kyle/projects/pokemon_diffusion/_testing_runs',
        'mixed_precision': 'fp16',
        'log_with': 'tensorboard',
        'data_dir': '/home/kyle/projects/pokemon_data/data'
    }

    model_parametrs = {
        'image_size': 64,
        'in_channels': 4,
        'out_channels': 4,
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
        'learning_rate': 2.5e-4,
        'lr_warmup_steps': 200,
        'save_image_epochs': 1,
        'save_model_epochs': 1,
        'seed': 42,
        'num_train_timesteps': 200, 
        'num_batches': None,
        'num_epochs': 1500,
        'dropout': 0.3,
        'shuffle': True
    }

    training_config = {**training_metadata, 
                       **model_parametrs, 
                       **training_hyperparameters}

    experiment = BaselineImageConditionalIADB(**training_config)
    experiment.run(num_processes=1)