import os

from datasets import Dataset, Image # Hugging Face Datasets
from torchvision.transforms import transforms, Compose, ToTensor, Normalize

DEFAULT_PREPROCESSING = Compose([
    ToTensor(), Normalize(mean=[0.5], std=[0.5])
])

def get_hf_img_dataset(path: str, preprocess: transforms=DEFAULT_PREPROCESSING):
    """
        Args:
        =====
        path: str
            Path to the directory containing the images.
        preprocess: torchvision.transforms.transforms
            Preprocessing pipeline to apply to the images.
    """
    img_paths = []
    for sprite in os.listdir(path):
        for png_file in os.listdir(os.path.join(path, sprite)):
            img_paths.append(os.path.join(path, sprite, png_file))

    dataset = Dataset.from_dict({"image": img_paths}).cast_column("image", Image())

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)