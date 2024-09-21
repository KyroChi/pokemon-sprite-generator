---
title: Pokémon Sprite Generator
emoji: 🐸
colorFrom: red
colorTo: blue
sdk: "gradio"
sdk_version: "4.43.0"
app_file: app.py
pinned: false
---

# Pokémon Sprite Generator

This project implements a conditional [iterative-$\alpha$ (de)blending](https://arxiv.org/pdf/2305.03486) model to generate Pokémon sprites. The project will continue to be updated as I gain access to better GPUs for training. Currently the generated images have the correct art style but don't resolve qualitative aspects which would allow them to be easily identified as Pokémon.

Additionally, the Pokémon dataset is relatively high-frequency; the edges are distinct from the body, transitions between colors are abrupt. Therefore I implemented [Fourier features](https://github.com/tancik/fourier-feature-networks) which I found to improve the output quality dramatically!

I scraped the dataset from Bulbapedia and the Pokémon API. My dataset creation code can be found [here](https://github.com/KyroChi/pokemon-sprite-dataset).

# Demo

This project is hosted as a HuggingFace Space [here](https://huggingface.co/spaces/krchickering/pokemon_generator). Below are static examples of Pokémon from the training set, and the outputs from my model.

![sample outputs](/resources/sample_output.png)

# Downloading and Running on your Machine
Clone the repo and run
```
conda env create -f environment.yml 
conda activate poke_sprite_generator
```
to set up the conda environment. Next run
```
python setup.py install
```
to install the package for this codebase.

For inference you can now run
```
python app.py
```
to launch a graio instance on your machine to play around with. Please see the app.py file for details about the implementation.

Finally, if you want to train you will need to download my dataset repo found [here](https://github.com/KyroChi), download the dataset, point the training experiments to the dataset, and then run any of the experiments in the experiment files. The demo was created using `python experiments/baseline_img_cond/baseline_img_cond_iadb_ff.py`, for example.
