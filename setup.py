import pathlib

import setuptools

_here = pathlib.Path(__file__).resolve().parent
print(_here)

name = "src"
author = ""
author_email = ""
description = (
    "Sparse function-space representation (SFR) of neural networks in PyTorch."
)

with open(_here / "README.md", "r") as f:
    readme = f.read()

url = ""

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
    "probabilistic-modelling",
    "bayesian-inference",
    "approximate-inference",
    "variational-inference",
    "bayesian-deep-learning",
    "bayesian-neural-networks",
    "gaussian-processes",
    "model-based-reinforcement-learning",
    "planning",
]

python_requires = ">=3.8"

install_requires = [
    "torch==2.0.0",
    "torchvision==0.15.1",
    "torchtyping==0.1.4",
    "matplotlib==3.5.1",
    "numpy==1.24.2",
    "ax-platform",
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
        "wandb",
        "hydra-core==1.3.2",
        "hydra-submitit-launcher==1.2.0",
        "jupyter==1.0.0",
        "netcal==1.3.5",  # needed for computing ECE
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
    # packages=setuptools.find_namespace_packages(),
    packages=setuptools.find_packages(exclude=["notebooks"]),
)
