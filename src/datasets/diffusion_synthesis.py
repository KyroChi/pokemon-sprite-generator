import csv
import os
import torch

from dataclasses import dataclass, asdict
from datasets import Dataset, Image
from torchvision.transforms import Resize, ToTensor
from typing import List

def reverse_dict(input_dict: dict):
    output_dict = {}
    for k, v in input_dict.items():
        output_dict[v] = k
    return output_dict

INT_TO_TYPE_MAP = {
    0: "normal",
    1: "fighting",
    2: "flying",
    3: "poison",
    4: "ground",
    5: "rock",
    6: "bug",
    7: "ghost",
    8: "steel",
    9: "fire",
    10: "water",
    11: "grass",
    12: "electric",
    13: "psychic",
    14: "ice",
    15: "dragon",
    16: "dark",
    17: "fairy",
}

TYPE_TO_INT_MAP = reverse_dict(INT_TO_TYPE_MAP)

INT_TO_COLOR_MAP = {
    0: "black",
    1: "blue",
    2: "brown",
    3: "gray",
    4: "green",
    5: "pink",
    6: "purple",
    7: "red",
    8: "white",
    9: "yellow",
}

COLOR_TO_INT_MAP = reverse_dict(INT_TO_COLOR_MAP)

INT_TO_SHAPE_MAP = {
    0: "head",
    1: "head and base",
    2: "serpentine",
    3: "head and legs",
    4: "quadruped",
    5: "tentacles",
    6: "insectoid",
    7: "head and arms",
    8: "wings",
    9: "bug wings",
    10: "fins",
    11: "bipedal",
    12: "bipedal with tail",
    13: "multiple_bodies"
}

SHAPE_TO_INT_MAP = reverse_dict(INT_TO_SHAPE_MAP)


def type_to_int(type: str) -> int:
    return TYPE_TO_INT_MAP[type]


def int_to_type(int_type: int) -> str:
    return INT_TO_TYPE_MAP[int_type]


def color_to_int(color: str) -> int:
    return COLOR_TO_INT_MAP[color]


def int_to_color(color: int) -> str:
    return INT_TO_COLOR_MAP[color]


def shape_to_int(shape: str) -> int:
    return SHAPE_TO_INT_MAP[shape]


def int_to_shape(shape: int) -> str:
    return INT_TO_SHAPE_MAP[shape]


def parse_sprite(sprite: str) -> tuple[bool, int]:
    chunks = sprite.split("_")
    try:
        pokemon_id = int(chunks[2])
    except Exception as _:
        pokemon_id = int(chunks[2][:-1])

    if len(chunks) == 4:
        is_shiny = True if chunks[3] == "s" else False
    elif len(chunks) == 5:
        is_shiny = True if chunks[4] == "s" else False
    else:
        is_shiny = False
    return is_shiny, pokemon_id


def prepare_png_image(image_path: str) -> Image:
    img = Image.open(image_path).convert("RGBA")
    img = Resize((96, 96))(img)
    return ToTensor()(img)


def list_from_types(types: str):
    # Input will be a string like "['normal', ' fire']". We have to
    # strip the brackets, split by comma, strip the quotes and
    # whitespace, and map to integers.
    return [type_to_int(t.strip()[1:-1]) for t in types[1:-1].split(",")]


@dataclass
class ConditionalData:
    name: str
    is_legendary: bool
    is_mythical: bool
    color: int
    shape: int
    types: List[int]
    is_shiny: bool

def embed(cond):
    vec = torch.zeros(7)

    vec[0] = 1. if cond['is_legendary'] else 0.5
    vec[1] = 1. if cond['is_mythical'] else 0.5
    vec[2] = (float(cond['color']) + 1.) / 10.
    vec[3] = float(cond['shape']) / 14.
    vec[4] = (float(cond['types'][0]) + 1.) / 18.
    vec[5] = (float(cond['types'][0]) + 1.) / 18. if len(cond['types']) == 1 else (float(cond['types'][1]) + 1.) / 18.
    vec[6] = 1. if cond['is_shiny'] else 0.5

    return vec

def get_unwrapped_sprite_dataset(path: str, samples_per_pokemon: int = None):
        img_paths = []
        for sprite in os.listdir(path):

            count = 0
            for png_file in os.listdir(os.path.join(path, sprite)):
                img_paths.append(os.path.join(path, sprite, png_file))

                count += 1
                if samples_per_pokemon is not None and count >= samples_per_pokemon:
                    break

        return Dataset.from_dict({"image": img_paths}).cast_column("image", Image())


def get_unwrapped_conditional_dataset(
        path: str, 
        get_shiny: bool = False,
        samples_per_pokemon: int = None
    ):
    assert(samples_per_pokemon is None or samples_per_pokemon > 0)

    img_paths = []
    conditions = []

    sprites = os.path.join(path, "gen_v_unwrapped_sprites")
    tabular = os.path.join(path, "poke_tabular.csv")
    get_shiny = get_shiny

    with open(tabular, "r") as f:
        poke_rows = list(csv.DictReader(f))
        
    for sprite in os.listdir(sprites):
        is_shiny, pokemon_id = parse_sprite(sprite)
        
        if not get_shiny and is_shiny:
            continue

        count = 0
        for png_file in os.listdir(os.path.join(sprites, sprite)):
            img_paths.append(os.path.join(sprites, sprite, png_file))
            conditions.append(
                embed(dict(
                    name=poke_rows[pokemon_id - 1]["name"],
                    is_legendary=not bool(poke_rows[pokemon_id - 1]["is_legendary"]),
                    is_mythical=not bool(poke_rows[pokemon_id - 1]["is_mythical"]),
                    color=color_to_int(poke_rows[pokemon_id - 1]["color"]),
                    shape=poke_rows[pokemon_id - 1]["shape"],
                    types=list_from_types(poke_rows[pokemon_id - 1]["types"]),
                    is_shiny=is_shiny,
                ))
            )

            count += 1
            if samples_per_pokemon is not None and count >= samples_per_pokemon:
                break

    return Dataset.from_dict({"image": img_paths, "conditions": conditions}).cast_column("image", Image())


def get_full_art_conditional_dataset(
        path: str,
    ):

    img_paths = []
    conditions = []

    full_size_path = os.path.join(path, "full_size")
    tabular = os.path.join(path, "poke_tabular.csv")

    with open(tabular, "r") as f:
        poke_rows = list(csv.DictReader(f))

    for pokemon in os.listdir(full_size_path):
        poke_id = int(pokemon.split("_")[0])
        img_paths.append(os.path.join(full_size_path, pokemon))
        conditions.append(
            embed(dict(
                name=poke_rows[poke_id - 1]["name"],
                is_legendary=not bool(poke_rows[poke_id - 1]["is_legendary"]),
                is_mythical=not bool(poke_rows[poke_id - 1]["is_mythical"]),
                color=color_to_int(poke_rows[poke_id - 1]["color"]),
                shape=poke_rows[poke_id - 1]["shape"],
                types=list_from_types(poke_rows[poke_id - 1]["types"]),
                is_shiny=False,
            ))
        )

    return Dataset.from_dict({"image": img_paths, "conditions": conditions}).cast_column("image", Image())


    