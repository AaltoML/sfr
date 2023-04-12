import pathlib

import setuptools

_here = pathlib.Path(__file__).resolve().parent
print(_here)

name = "src"
author = "Aidan Scannell"
author_email = "scannell.aidan@gmail.com"
description = "Neural sparse GPs for prediction and fast updates PyTorch."

with open(_here / "README.md", "r") as f:
    readme = f.read()

url = "https://github.com/aaltoml/bnn-to-dual-svgp"

license = "Apache-2.0"

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Intended Audience :: Information Technology",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
]
keywords = [
    "deep-learning",
    "machine-learning",
    "bayesian-deep-learning",
    "gaussian-processes",
    "variational-inference",
    "model-based-reinforcement-learning",
    "bayesian-inference",
    "planning",
]

python_requires = ">=3.8"

install_requires = [
    "matplotlib==3.7.1",
    "numpy==1.24.2",
    "torch==2.0.0",
    # functorch  # needed for vmap
    "torchtyping==0.1.4",
    # laplace-torch
    "gpytorch==1.9.1",
    # git+https://github.com/pytorch/rl
    # "torchrl==0.1.0",  # using this on triton
    # "./src/third_party/rl-0.1.0", # using this locally
    f"torchrl @ file://{_here}/src/third_party/rl-0.1.0",
]
extras_require = {
    "dev": [
        "black==23.3.0",
        "pre-commit==3.2.2",
        "pyright==1.1.301",
        "isort==5.12.0",
        "pyflakes==3.0.1",
        "pytest==7.2.2",
    ],
    "experiments": [
        "wandb==0.14.1",
        "hydra-core==1.3.2",
        "hydra-submitit-launcher==1.2.0",
        "mujoco==2.3.3",
        "dm_control==1.0.11",  # deepmind control suite
        "opencv-python==4.7.0.72",
        "moviepy==1.0.3",  # rendering
        "tikzplotlib==0.10.1",
        "tabulate==0.9.0",
    ],
}

setuptools.setup(
    name=name,
    version="0.1.0",
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    keywords=keywords,
    long_description=readme,
    long_description_content_type="text/markdown",
    url=url,
    license=license,
    classifiers=classifiers,
    zip_safe=False,
    python_requires=python_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    packages=setuptools.find_namespace_packages(),
)
