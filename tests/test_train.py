import json

import pytest
import torch

from minselfflow.train import Dist, parse_config, train


@pytest.mark.parametrize("method", ["flow", "dual", "selfflow"])
def test_train_smoke_and_resume(tmp_path, method):
    images = torch.randint(0, 256, (64, 3, 32, 32), dtype=torch.uint8)
    labels = torch.randint(0, 10, (64,))
    args = ["--out_dir", str(tmp_path), "--dim", "32", "--depth", "3", "--heads", "2", "--batch_size", "8"]
    args += ["--log_every", "2", "--sample_every", "4", "--ckpt_every", "4", "--compile", "false", "--method", method]
    d = Dist(0, 1, torch.device("cpu"))

    train(parse_config([*args, "--steps", "4"]), images, labels, d)
    assert (tmp_path / "samples_000004.png").exists()
    assert torch.load(tmp_path / "ckpt.pt")["step"] == 4

    train(parse_config([*args, "--steps", "6"]), images, labels, d)  # resumes from step 4
    steps = [json.loads(line)["step"] for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert steps == [2, 4, 6]
