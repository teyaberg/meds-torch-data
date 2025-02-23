#!/usr/bin/env python

import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from . import ETL_CFG, MAIN_CFG, RUNNER_CFG
from .commands import run_command

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=str(MAIN_CFG.parent), config_name=MAIN_CFG.stem)
def main(cfg: DictConfig):
    """Runs the end-to-end MEDS Extraction pipeline."""

    MEDS_dataset_dir = Path(cfg.MEDS_dataset_dir)
    output_dir = Path(cfg.output_dir)
    stage_runner_fp = cfg.get("stage_runner_fp", None)

    # Then we construct the rest of the command
    command_parts = [
        f"INPUT_DIR={str(MEDS_dataset_dir.resolve())}",
        f"OUTPUT_DIR={str(output_dir.resolve())}",
        "MEDS_transform-runner",
        f"--config-path={str(RUNNER_CFG.parent.resolve())}",
        f"--config-name={RUNNER_CFG.stem}",
        f"pipeline_config_fp={str(ETL_CFG.resolve())}",
    ]
    if int(os.getenv("N_WORKERS", 1)) <= 1:
        logger.info("Running in serial mode as N_WORKERS is not set.")
        command_parts.append("~parallelize")

    if stage_runner_fp:
        command_parts.append(f"stage_runner_fp={stage_runner_fp}")

    command_parts.append("'hydra.searchpath=[pkg://MEDS_transforms.configs]'")
    run_command(command_parts, cfg)


if __name__ == "__main__":
    main()
