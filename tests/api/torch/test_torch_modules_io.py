from pathlib import Path

import torch

from pyTorchAutoForge.api.torch import AutoForgeModuleSaveMode, SaveModel


def test_SaveModel_accepts_torch_device(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    model_path = tmp_path / "linear_model"

    SaveModel(
        model=model,
        model_filename=model_path,
        save_mode=AutoForgeModuleSaveMode.MODEL_STATE_DICT,
        target_device=torch.device("cpu"),
    )

    assert (tmp_path / "linear_model_statedict.pth").is_file()
