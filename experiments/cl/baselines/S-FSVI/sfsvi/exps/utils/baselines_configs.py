"""Configs for baselines."""
from typing import Dict, List

from baselines.vcl.run_vcl import add_vcl_args
from sfsvi.general_utils.log import EXPS_ROOT, PROJECT_ROOT
from sfsvi.exps.utils.config_template import ConfigTemplate

CL_ROOT = PROJECT_ROOT / "fsvi"
BASELINE_ROOT = EXPS_ROOT / "baselines"


VCL_TEMPLATE = ConfigTemplate(add_args_fn=add_vcl_args)
VCL_RUNFILE = PROJECT_ROOT / "baselines/vcl/run_vcl.py"
VCL_PREFIX = f"python {VCL_RUNFILE}"

def get_vcl_permuted_MNIST_with_coreset_config() -> Dict:
    cmd_str = """
        --dataset pmnist
        --batch_size 256    --n_epochs 100
        --hidden_size 256  --n_layers 2
        --seed 42     
        --n_coreset_inputs_per_task 200
        --select_method random_choice     
        """
    return VCL_TEMPLATE.parse_args(cmd_str, template_info=True)


def get_vcl_split_MNIST_with_coreset_config() -> Dict:
    cmd_str = """
        --dataset smnist
        --n_epochs 120
        --hidden_size 256  --n_layers 2  
        --seed 42     
        --n_coreset_inputs_per_task 40
        --select_method random_choice     
        """
    return VCL_TEMPLATE.parse_args(cmd_str, template_info=True)


def baseline_configs_to_file(
    configs: List[Dict],
    subdir: str,
    folder: str = "jobs",
    root=BASELINE_ROOT,
    runner_version="v1",
):
    vcl_configs = [c for c in configs if "vcl" in c["template"]]
    VCL_TEMPLATE.configs_to_file(
        configs=vcl_configs,
        file_path=root / folder / f"{subdir}.sh",
        prefix=VCL_PREFIX,
        mode="a",
    )
