# Pokémon Sprite Generator

This project implements a conditional [iterative-$\alpha$ (de)blending](https://arxiv.org/pdf/2305.03486) model to generate Pokémon sprites. Currently, the generated images reflect the correct art style, but don't resolve qualitative aspects which would allow them to be easily identified as Pokémon, such as distinct faces. We will continue to update this project as we gain increased GPU access and perform further training.

Additionally, the Pokémon dataset is relatively high-frequency; the edges are distinct from the body, and transitions between colors are abrupt. To address this, we implemented [Fourier features](https://github.com/tancik/fourier-feature-networks) which we found to improve the output quality dramatically.

We scraped the dataset from Bulbapedia and the Pokémon API. Our dataset creation code can be found [here](https://github.com/KyroChi/pokemon-sprite-dataset).

# Demo

This project is hosted as a HuggingFace Space [here](https://huggingface.co/spaces/krchickering/pokemon_generator). Below are static examples of Pokémon sprite outputs from the model.

![sample outputs](/resources/sample_output.png)

# Downloading and Running on your Machine
To download and run on your machine, first clone the repository and run
```
conda env create -f environment.yml 
conda activate poke_sprite_generator
```
to set up the conda environment. Next run
```
git-lfs pull
python setup.py install
```
to download the checkpoint files and install the package for this codebase.

For inference, you can now run
```
python app.py
```
to launch a gradio instance on your machine to play around with. Please see the app.py file for details about the implementation.

Finally, if you want to train the model you will need to download our dataset repo found [here](https://github.com/KyroChi), download the dataset, point the training experiments to the dataset, and then run any of the experiments in the experiment files. For example, the demo was created using `python experiments/baseline_img_cond/baseline_img_cond_iadb_ff.py`.

