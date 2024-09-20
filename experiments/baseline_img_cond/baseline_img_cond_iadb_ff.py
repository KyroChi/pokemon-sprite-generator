"""Baseline conditional image model
- Baseline
- Conditional
- Classifier-free guided diffusion
- IADB diffusion
- Fourier features
"""

import os
import random
import torch
import torch.nn.functional as F

from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.utils import save_image

from src.common.utils import RandomCenteredCrop, feature_mapping
from src.experiment.experiment import Experiment
from src.datasets.diffusion_synthesis import get_unwrapped_conditional_dataset, get_full_art_conditional_dataset

@torch.no_grad()
def sample_iadb(model, x0, cond, nb_step, w, feature_mapping):
    model.eval()
    x_alpha = x0

    cond = cond.to(x0.device)
    zero = torch.zeros_like(cond).to(x0.device)

    for t in range(nb_step):
        alpha_start = (t/nb_step)
        alpha_end =((t+1)/nb_step)

        fourier = feature_mapping(x_alpha)

        net_input_cond = torch.cat([x_alpha, fourier, cond], dim=1)
        net_input_uncond = torch.cat([x_alpha, fourier, zero], dim=1)

        d_cond = model(net_input_cond, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        d_uncond = model(net_input_uncond, torch.tensor(alpha_start, device=x_alpha.device))['sample']

        d = (1 + w) * d_cond - w * d_uncond

        x_alpha = x_alpha + (alpha_end - alpha_start) * d

    model.train()
    return x_alpha

def evaluate(config, model, epoch, x0_eval, cond_eval, feature_mapping, w_interpolants=[0, 1, 2, 3]):
    steps = config.num_inference_timesteps

    for w in w_interpolants:
        sample = sample_iadb(
            model=model, 
            x0=x0_eval, 
            cond=cond_eval,
            nb_step=steps, 
            w=w,
            feature_mapping=feature_mapping
        )
        sample = 0.5 * sample + 0.5
        dirname = os.path.join(config.project_dir, 'output')
        os.makedirs(dirname, exist_ok=True)
        save_image(sample, f'{dirname}/export_{str(epoch).zfill(8)}_w_{w}.png')

class BaselineImageIADB(Experiment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # unrwapped sprite dataset, balanced number of samples per pokemon
        # self.dataset = get_unwrapped_conditional_dataset(
        #     path=self.config.data_dir, 
        #     get_shiny=True,
        #     samples_per_pokemon=30
        # )

        # full art dataset
        self.dataset = get_full_art_conditional_dataset(self.config.data_dir)

        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.config.image_size, self.config.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def transform(examples):
            images = [preprocess(image.convert("RGBA")) for image in examples["image"]]
            return {"images": images, "conditions": torch.Tensor(examples["conditions"])}

        self.dataset.set_transform(transform)

        self.train_dataloader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.config.train_batch_size, 
            shuffle=self.config.shuffle,
            num_workers=4,            
        )

        # Set up model, optimizer, scheduler, etc.
        self.model = UNet2DModel(
            sample_size=self.config.image_size,
            in_channels=self.config.in_channels + 2 * self.config.ff_num_features + 7,
            out_channels=self.config.out_channels,
            layers_per_block=self.config.layers_per_block,
            block_out_channels=[block[1] for block in self.config.down_blocks],
            down_block_types=[block[0] for block in self.config.down_blocks],
            up_block_types=self.config.up_blocks,
            dropout=self.config.dropout,
        )

        
        # model_config = load_file('/home/kyle/projects/pokemon_diffusion/_testing_runs/baseline_img_cond_iadb_ff/2024-09-18_07-40-38/models/model_6500.pt/model.safetensors')
        # conv_in_weight = model_config['conv_in.weight']
        # randn_new_features = 0.001 * torch.ones(conv_in_weight.shape[0], 7, conv_in_weight.shape[2], conv_in_weight.shape[3])
        # model_config['conv_in.weight'] = torch.cat([conv_in_weight, randn_new_features], dim=1)

        # self.model.load_state_dict(
        #     model_config,
        #     strict=False # Top layers of the encoder will be the wrong size
        # )

        # model_config = load_file('/home/kyle/projects/pokemon_diffusion/_testing_runs/baseline_img_cond_iadb_ff/2024-09-13_13-31-25/models/model_150.pt/model.safetensors')

        # self.model.load_state_dict(model_config)

        # for _, param in self.model.named_parameters():
        #     param.requires_grad = False

        # for _, param in self.model.mid_block.named_parameters():
        #     param.requires_grad = True

        # # for _, param in self.model.down_blocks[-1].named_parameters():
        # #     param.requires_grad = True

        # for _, param in self.model.down_blocks[-2].named_parameters():
        #     param.requires_grad = True

        # # for _, param in self.model.up_blocks[-1].named_parameters():
        # #     param.requires_grad = True

        # for _, param in self.model.up_blocks[-2].named_parameters():
        #     param.requires_grad = True


        # # Re-train only the attention layers:
        # for _, param in self.model.named_parameters():
        #     param.requires_grad = False

        # attn_down_blocks = [block for block in self.model.down_blocks if 'Attn' in block._get_name()]
        # attn_up_blocks = [block for block in self.model.up_blocks if 'Attn' in block._get_name()]

        # for block in attn_down_blocks + attn_up_blocks:
        #     for _, param in block.named_parameters():
        #         param.requires_grad = True


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

        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        mean = torch.full((self.config.ff_num_features, self.config.in_channels), 0.0)
        std = torch.full((self.config.ff_num_features, self.config.in_channels), self.config.ff_scale)

        B = torch.normal(mean, std)

        torch.save(B, os.path.join(self.config.project_dir, "B_tensor.pt"))

        B = B.to(self.accelerator.device)

        self.feature_mapping = lambda x: feature_mapping(x, B)

    def train_single_iteration(self, idx, batch, *args, **kwargs):
        x1 = batch["images"] # Images are assumed to already be normalized
        x0 = torch.randn_like(x1).to(x1.device)
        cond = batch["conditions"]

        bs, _, h, w = x1.shape
        cond = cond.to(x1.device)
        cond = cond.view(bs, -1).unsqueeze(-1).unsqueeze(-1).expand(bs, -1, h, w)

        if self.x0_eval is None:
            self.x0_eval = x0
            self.cond_eval = cond

        # # Randomly mask out coordinates in the condition. This is a different method than Ho et. al.
        # cond = torch.where(torch.rand_like(cond) < 0.3, torch.zeros_like(cond), cond)

        if random.random() < self.config.cfg_dropout:
            cond = torch.zeros_like(cond)
        
        alpha = torch.rand(bs, device=x1.device)
        x_alpha = torch.lerp(x0, x1, alpha.view(-1, 1, 1, 1))

        fourier = self.feature_mapping(x_alpha)
        x_alpha = torch.cat([x_alpha, fourier], dim=1)

        x_alpha = torch.cat([x_alpha, cond], dim=1)

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
        if (epoch + 1) % (self.config.save_image_epochs) == 0 or epoch == self.config.num_epochs - 1:
            evaluate(self.config, self.model, epoch + 1, self.x0_eval, self.cond_eval, self.feature_mapping, w_interpolants=[0, 2])

        if (epoch + 1) % self.config.save_model_epochs == 0 or epoch == self.config.num_epochs - 1:
            self.accelerator.save_model(self.model, os.path.join(self.config.project_dir, "models", f"model_{epoch + 1}.pt"))

if __name__ == "__main__":
    training_metadata = {
        'experiment_name': 'baseline_img_cond_iadb_ff',
        'project_dir': '/home/kyle/projects/pokemon_diffusion/_testing_runs',
        'mixed_precision': 'fp16',
        'log_with': 'tensorboard',
        'data_dir': '/home/kyle/projects/pokemon_data/data'
    }

    model_parametrs = {
        'image_size': 256,
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
        'train_batch_size': 32,
        'eval_batch_size': 16,
        'gradient_accumulation_steps': 1,
        'learning_rate': 1e-4,
        'lr_warmup_steps': 200,
        'save_image_epochs': 20,
        'save_model_epochs': 20,
        'dropout': 0.3,
        'seed': 42,
        'num_train_timesteps': 200, 
        'num_batches': None,
        'num_epochs': 6500,
        'shuffle': True,
        'ff_num_features': 16,
        'ff_scale': 2.5,
        'cfg_dropout': 0.1,
    }

    training_config = {**training_metadata, 
                       **model_parametrs, 
                       **training_hyperparameters}

    experiment = BaselineImageIADB(**training_config)
    experiment.run(num_processes=1)