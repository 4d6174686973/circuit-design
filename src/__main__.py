from omegaconf import DictConfig
import hydra
import os

from src.setup import setup_and_train_qcbm

# with version_base=none in hydra.main, it will not create nice folder structure
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # Create output directories
    os.makedirs("mps", exist_ok=True)
    os.makedirs("qcbm", exist_ok=True)

    # Setup and train QCBM
    setup_and_train_qcbm(cfg)


if __name__ == "__main__":
    main()