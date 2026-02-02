import os
import optuna
from torch import optim
import torch

from config import Config, make_run_dir, to_serializable, save_yaml, save_json
from val import hold_out_train_and_validate

OPTIM_MAP = {
    "AdamW": optim.AdamW,
    "Adam": optim.Adam,
    "SGD": optim.SGD,
}

def objective(trial: optuna.Trial) -> float:
    cfg = Config()
    cfg.epochs = 2
    cfg.batch_size = 25
    cfg.optimizer_params['lr'] = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
    cfg.optimizer_params['weight_decay'] = trial.suggest_float("weight_decay", 1e-5, 1e-3,  log=True)
    cfg.optimizer_cls = OPTIM_MAP[trial.suggest_categorical("optimizer", ["AdamW", "Adam"])]
    cfg.model_name = trial.suggest_categorical("model_name", ["efficientnet_b4", "efficientnet_b5"])
    cfg.drop_rate = trial.suggest_float("drop_rate", 0.0, 0.3)
    cfg.drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 0.3)

    run_dir = make_run_dir("runs", trial)
    cfg.save_path = os.path.join(run_dir, "best_model.pth")

    # -----------------------
    # (1) 전체 Config 저장
    # -----------------------
    cfg_dict = to_serializable(cfg)
    save_yaml(f"{run_dir}/config.yaml", cfg_dict)

    # -----------------------
    # (2) trial 파라미터 저장
    # -----------------------
    save_json(f"{run_dir}/trial.json", {
        "trial_number": trial.number,
        "params": trial.params
    })

    # -----------------------
    # Optuna 내부에도 저장
    # -----------------------
    trial.set_user_attr("full_config", cfg_dict)

    try:
        return hold_out_train_and_validate(cfg, trial)[1]
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        raise

if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=0,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name="deepfake_effnet",
        direction="maximize",
        storage="sqlite:///optuna_deepfake.db",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=25)