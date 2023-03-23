import os
import shutil
from pathlib import Path
from typing import List

FILE_DIR = os.path.dirname(__file__)
KNOWLEDGE_GEN_DIR = "reference/generated/knowledge_base"
TUTORIALS_GEN_DIR = "reference/generated/tutorials"
# TUTORIALS_SRC_GEN_DIR = "reference/generated/tutorials/src"
TUTORIALS_MEDIA_GEN_DIR = "reference/generated/tutorials/media"


def _get_file_content(name: str) -> List[str]:
    """A function to get the content of a reference file.

    Given the name of a knowledge base file, populates a file template. The result can be used to link a knowledge base
    entry to the Sphinx docs.

    Args:
        name (str): name of the file to be referenced (without extension).

    Returns: List of strings

    """
    return [
        "..\n",
        "   This file is generated by knowledge_base.py, manual changes will be overwritten.\n",
        "\n",
        f".. include:: ../../../../../knowledge_base/{name}.md\n",
        "   :parser: myst_parser.sphinx_\n",
        "\n",
    ]


def generate_knowledge_base_references(knowledge_base_path: str) -> None:
    """Creates a reference file per knowledge base entry.

    Sphinx natively doesn't support adding files from outside its root directory. To include the knowledge base in
    our docs (https://pytorch.org/rl/) each entry is linked using an auto-generated file that references the original.

    Args:
        knowledge_base_path (str): path to the knowledge base.
    """
    # Create target dir
    target_path = os.path.join(FILE_DIR, KNOWLEDGE_GEN_DIR)
    Path(target_path).mkdir(parents=True, exist_ok=True)

    # Iterate knowledge base files
    file_paths = os.listdir(os.path.join(FILE_DIR, knowledge_base_path))
    for file_path in file_paths:
        name = Path(file_path).stem

        # Skip README, it is already included in `knowledge_base.rst`
        if name == "README":
            continue

        # Existing files will be overwritten in 'w' mode
        with open(os.path.join(target_path, f"{name}.rst"), "w") as file:
            file.writelines(_get_file_content(name))


def generate_tutorial_references(tutorial_path: str, file_type: str) -> None:
    """Creates a python file per tutorial script.

    Sphinx natively doesn't support adding files from outside its root directory. To include the tutorials in
    our docs (https://pytorch.org/rl/) each entry is locally copied.

    Args:
        tutorial_path (str): path to the tutorial scripts.
    """
    # Create target dir
    if file_type == "tutorial":
        target_path = os.path.join(FILE_DIR, TUTORIALS_GEN_DIR)
    # elif file_type == "src":
    #     target_path = os.path.join(FILE_DIR, TUTORIALS_SRC_GEN_DIR)
    else:
        target_path = os.path.join(FILE_DIR, TUTORIALS_MEDIA_GEN_DIR)
    Path(target_path).mkdir(parents=True, exist_ok=True)

    # Iterate tutorial files and copy
    file_paths = [
        os.path.join(tutorial_path, f)
        for f in os.listdir(tutorial_path)
        if f.endswith((".py", ".rst", ".png"))
    ]

    for file_path in file_paths:
        shutil.copyfile(file_path, os.path.join(target_path, Path(file_path).name))