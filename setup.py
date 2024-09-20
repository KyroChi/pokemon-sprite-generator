from setuptools import setup

setup(
    name='pokemon_diffusion',
    packages=['src',
              'src.common', 
              'src.datasets',
              'src.experiment',
              'src.metrics',
              'src.models'],
)