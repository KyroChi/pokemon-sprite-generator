import gradio as gr
import numpy as np
import torch
import torchvision.transforms as tf

from diffusers import UNet2DModel
from safetensors.torch import load_file
from torchvision.utils import make_grid

from src.datasets.diffusion_synthesis import int_to_color, int_to_shape, int_to_type, embed, color_to_int, shape_to_int, type_to_int

config = torch.load('resources/config.pt')
B = torch.load('resources/B_tensor.pt').to("cuda")
model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=config.in_channels + 2 * config.ff_num_features + 7,
    out_channels=config.out_channels,
    layers_per_block=config.layers_per_block,
    block_out_channels=[block[1] for block in config.down_blocks],
    down_block_types=[block[0] for block in config.down_blocks],
    up_block_types=config.up_blocks,
    dropout=config.dropout
)

model.load_state_dict(
    load_file('resources/model.safetensors')
)

def feature_mapping(x):
    x_proj = torch.einsum("fc, Bchw -> Bfhw", B, (2 * np.pi * x))
    x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)
    return x_proj

@torch.no_grad()
def cfg_sample_iadb(model, x0, cond, nb_steps, w):
    model.eval()
    x_alpha = x0

    cond = cond.to(x0.device)
    zero = torch.zeros_like(cond).to(x0.device)

    for t in range(nb_steps):
        alpha_start = (t/nb_steps)
        alpha_end =((t+1)/nb_steps)

        fourier = feature_mapping(x_alpha)
        x_alpha = torch.cat([x_alpha, fourier], dim=1)

        net_input_cond = torch.cat([x_alpha, cond], dim=1)
        net_input_uncond = torch.cat([x_alpha, zero], dim=1)

        d_cond = model(net_input_cond, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        d_uncond = model(net_input_uncond, torch.tensor(alpha_start, device=x_alpha.device))['sample']

        d = (1 + w) * d_cond - w * d_uncond

        x_alpha = x_alpha[:, 0:4, :, :] + (alpha_end - alpha_start) * d

    model.train()
    return x_alpha

@torch.no_grad()
def generate_samples(
    input_dict
):
    input_dict['types'] = [
        type_to_int(input_dict['type1']) + 1, type_to_int(input_dict['type2']) + 1
    ]
    input_dict['color'] = color_to_int(input_dict['color']) + 1
    input_dict['shape'] = shape_to_int(input_dict['shape']) + 1
    
    cond_vec = embed(input_dict)

    image_size = input_dict['image_size']

    b = 16 if input_dict['image_size'] < 200 else 4

    x0 = torch.randn(b, 4, image_size, image_size)
    cond_vec = cond_vec.view(7, -1).unsqueeze(0).unsqueeze(-1).expand(b, -1, image_size, image_size)

    print(cond_vec.shape)

    x_alpha = cfg_sample_iadb(
        model.to("cuda"), 
        x0.to("cuda"), 
        cond_vec.to("cuda"), 
        input_dict['inference_timesteps'], 
        input_dict['inverse_temperature']
    )

    x_alpha = make_grid(x_alpha, nrow=int(np.sqrt(b)), padding=1)
    # x_alpha = tf.Resize((1024, 1024), interpolation=tf.InterpolationMode.NEAREST)(x_alpha)

    print(x_alpha.min(), x_alpha.max())
    x_alpha = (0.5 * x_alpha + 0.5).clamp_(-1, 1)
    # x_alpha = (x_alpha - x_alpha.min()) / (x_alpha.max() - x_alpha.min())
    print(x_alpha.min(), x_alpha.max())

    # x_alpha.mul(255).add_(0.5).clamp_(0, 255)
    x_alpha = x_alpha.cpu().permute(1, 2, 0).numpy()

    return x_alpha
    

# Define the function that processes the inputs
def process_inputs(type1, type2, color, shape, inverse_temperature, image_size, inference_timesteps): 
    # Here you can add the logic to process the inputs
    # For demonstration, we will just return the inputs as a dictionary
    input_dict = {
        "type1": type1.lower(),
        "type2": type2.lower(),
        "is_legendary": False,
        "is_mythical": False,
        "color": color.lower(),
        "shape": shape.lower(),
        "is_shiny": False,
        "inverse_temperature": inverse_temperature,
        "image_size": image_size,
        "inference_timesteps": int(inference_timesteps)
    }

    return generate_samples(input_dict)

# Define the input widgets
type1_input = gr.Dropdown(choices=[int_to_type(i).capitalize() for i in range(18)], label="Type 1", value="Fire")
type2_input = gr.Dropdown(choices=[int_to_type(i).capitalize() for i in range(18)], label="Type 2", value="Dark")
color_input = gr.Dropdown(choices=[int_to_color(i).capitalize() for i in range(10)], label="Color", value="Red")
shape_input = gr.Dropdown(choices=[int_to_shape(i).capitalize() for i in range(14)], label="Shape", value="Quadruped")
inverse_temperature_input = gr.Slider(minimum=-1, maximum=10, value=4.5, label="CFG Parameter")
input_image_size = gr.Dropdown(choices=[32, 64, 96, 256], label="Image Size", value=256)
inference_timesteps = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Inference Timesteps")

# Open the file resources/app_text.txt and put the contents into description as a string
with open('resources/app_text.txt', 'r') as f:
    description = f.read()

# Create the Gradio interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        type1_input,
        type2_input,
        color_input,
        shape_input,
        inverse_temperature_input,
        input_image_size,
        inference_timesteps
    ],
    outputs=gr.Image(type="numpy"),
    title="Pokémon Conditional Diffusion Model",
    description=description
)

# Launch the Gradio interface
iface.launch(share=True)