"""
Optuna Hyperparameter Optimization (Round 1 Val 전용).

사용 예:
    python3 hpo.py                    # 새 Study 시작 (기존 storage 있으면 이어서 resume)
    python3 hpo.py --n-trials 50      # n_trials 덮어쓰기
    python3 hpo.py --export-best      # 현재 study의 best를 best_config.yaml로 저장
    python3 hpo.py --dry-run          # 파라미터 샘플링만 찍고 종료 (디버그)

설계 원칙:
  - Round 1의 val set에서만 탐색 → 이후 Round의 test 기간이 HPO에 영향 없음.
  - HyperbandPruner로 성능 낮은 trial 조기 종료.
  - SQLite storage로 중단/재개 지원.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from HMM import train_market_hmm
from config import load_config, parse_net_arch
from rolling_window import generate_rounds, slice_with_context
from train import build_ppo, evaluate_deterministic, make_env


# ─────────────────────────────────────────────────────────────────
# Optuna용 Callback (pruning 지원)
# ─────────────────────────────────────────────────────────────────

class OptunaValidationCallback(BaseCallback):
    """
    ValidationCallback + Optuna 보고/pruning.
    eval_freq 스텝마다 val DSR 계산 → trial.report → should_prune 확인.
    """

    def __init__(
        self,
        val_env,
        eval_freq: int,
        trial: optuna.Trial,
        warmup_steps: int,
    ):
        super().__init__(verbose=0)
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.trial = trial
        self.warmup_steps = warmup_steps
        self.best_val_dsr = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            dsr_sum, _ = evaluate_deterministic(self.model, self.val_env)
            if dsr_sum > self.best_val_dsr:
                self.best_val_dsr = float(dsr_sum)

            # Optuna에 중간 결과 보고 (running best 기준)
            self.trial.report(self.best_val_dsr, self.n_calls)

            if (
                self.n_calls >= self.warmup_steps
                and self.trial.should_prune()
            ):
                raise optuna.TrialPruned()
        return True


# ─────────────────────────────────────────────────────────────────
# 파라미터 샘플링
# ─────────────────────────────────────────────────────────────────

def suggest_params(trial: optuna.Trial, space: dict) -> dict:
    """config.hpo.param_space를 Optuna suggest_* 호출로 변환."""
    out = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "loguniform":
            out[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=True
            )
        elif t == "uniform":
            out[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif t == "categorical":
            out[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"알 수 없는 param_space type: {t}")
    return out


def apply_suggested(base_ppo: dict, suggested: dict) -> dict:
    """base ppo config에 suggested 값 덮어씀. net_arch는 문자열 → 리스트."""
    merged = dict(base_ppo)
    for k, v in suggested.items():
        merged[k] = v
    if "net_arch" in merged and isinstance(merged["net_arch"], str):
        merged["net_arch"] = parse_net_arch(merged["net_arch"])
    return merged


# ─────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────

def make_objective(cfg: dict, df: pd.DataFrame, round_spec):
    """
    Optuna objective factory. Round 1 데이터와 base cfg를 closure로 캡처.
    """
    hpo = cfg["hpo"]
    window = cfg["data"]["window"]
    n_regimes = cfg["data"]["n_regimes"]

    train_df = df.loc[round_spec.train_start : round_spec.train_end]
    val_df = slice_with_context(df, round_spec.val_start, round_spec.val_end, window)

    # HMM은 trial 무관 (train data로 한 번만 학습 → 모든 trial에서 공유)
    print(f"[HPO] HMM 학습 중 (Round {round_spec.round_index + 1} train)...")
    hmm_model, hmm_scaler = train_market_hmm(train_df, n_regimes=n_regimes)

    def objective(trial: optuna.Trial) -> float:
        suggested = suggest_params(trial, hpo["param_space"])

        # trial용 cfg 복사 (원본 수정 금지)
        trial_cfg = copy.deepcopy(cfg)
        trial_cfg["ppo"] = apply_suggested(trial_cfg["ppo"], suggested)

        dsrs = []
        for seed in range(hpo["n_seeds_per_trial"]):
            set_random_seed(seed)
            train_env = make_env(train_df, trial_cfg, hmm_model, hmm_scaler)
            val_env = make_env(val_df, trial_cfg, hmm_model, hmm_scaler)
            model = build_ppo(train_env, seed, trial_cfg, prev_params=None)

            callback = OptunaValidationCallback(
                val_env=val_env,
                eval_freq=trial_cfg["train"]["eval_freq"],
                trial=trial,
                warmup_steps=hpo["pruner_warmup_steps"],
            )
            model.learn(
                total_timesteps=hpo["total_timesteps"],
                callback=callback,
                progress_bar=False,
            )
            dsrs.append(callback.best_val_dsr)

        return float(np.mean(dsrs))

    return objective


# ─────────────────────────────────────────────────────────────────
# Pruner / Study 헬퍼
# ─────────────────────────────────────────────────────────────────

def build_pruner(hpo_cfg: dict) -> optuna.pruners.BasePruner:
    name = hpo_cfg.get("pruner", "hyperband").lower()
    if name == "hyperband":
        return HyperbandPruner(
            min_resource=hpo_cfg["pruner_warmup_steps"],
            max_resource=hpo_cfg["total_timesteps"],
        )
    if name == "median":
        return MedianPruner(n_warmup_steps=hpo_cfg["pruner_warmup_steps"])
    if name == "none":
        return NopPruner()
    raise ValueError(f"알 수 없는 pruner: {name}")


def create_or_load_study(cfg: dict) -> optuna.Study:
    hpo = cfg["hpo"]
    storage = hpo["storage"]

    # SQLite 파일 디렉토리 확보
    if storage.startswith("sqlite:///"):
        sqlite_path = Path(storage[len("sqlite:///"):])
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    return optuna.create_study(
        study_name=hpo["study_name"],
        storage=storage,
        direction="maximize",
        pruner=build_pruner(hpo),
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )


# ─────────────────────────────────────────────────────────────────
# Export best → YAML
# ─────────────────────────────────────────────────────────────────

def export_best_config(cfg: dict) -> Path:
    study = create_or_load_study(cfg)
    if not study.best_trial:
        raise RuntimeError("완료된 trial이 없습니다.")

    best_ppo = apply_suggested(cfg["ppo"], study.best_params)
    # categorical로 넘어온 net_arch가 문자열이면 리스트로 (이미 apply_suggested에서 처리됨)

    out_path = Path(cfg["paths"]["hpo_dir"]) / "best_config.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(
            {
                "ppo": best_ppo,
                "_hpo_meta": {
                    "best_value": float(study.best_value),
                    "best_trial_number": int(study.best_trial.number),
                    "n_trials_done": len(study.trials),
                    "study_name": cfg["hpo"]["study_name"],
                },
            },
            f,
            sort_keys=False,
        )
    return out_path


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--export-best", action="store_true",
                        help="현재 study의 best를 YAML로 저장 후 종료")
    parser.add_argument("--dry-run", action="store_true",
                        help="파라미터 샘플링만 출력하고 종료")
    args = parser.parse_args()

    cfg = load_config(args.config)
    hpo = cfg["hpo"]
    n_trials = args.n_trials or hpo["n_trials"]

    if args.export_best:
        out = export_best_config(cfg)
        print(f"best_config 저장: {out}")
        return

    df = pd.read_csv(
        cfg["paths"]["data_csv"], index_col=0, parse_dates=True
    ).sort_index()
    rounds = generate_rounds(df, cfg)
    target_round = rounds[hpo["round_index"]]

    print("=" * 70)
    print(" Optuna HPO")
    print("=" * 70)
    print(f"  대상: {target_round.describe()}")
    print(f"  trials: {n_trials} (n_seeds_per_trial={hpo['n_seeds_per_trial']})")
    print(f"  timesteps/trial: {hpo['total_timesteps']:,}")
    print(f"  pruner: {hpo['pruner']}")
    print(f"  storage: {hpo['storage']}")

    study = create_or_load_study(cfg)
    print(f"  study: '{study.study_name}' (기존 trials={len(study.trials)})")

    if args.dry_run:
        trial = study.ask()
        sugg = suggest_params(trial, hpo["param_space"])
        print("\n[dry-run] 샘플링 결과:")
        for k, v in sugg.items():
            print(f"  {k}: {v}")
        return

    objective = make_objective(cfg, df, target_round)
    study.optimize(
        objective,
        n_trials=n_trials,
        catch=(ValueError,),  # 수치 불안정 trial은 실패로 처리
        show_progress_bar=False,
    )

    print("\n" + "=" * 70)
    print(" HPO 완료")
    print("=" * 70)
    print(f"  최고 value: {study.best_value:.4f}")
    print(f"  최고 trial: #{study.best_trial.number}")
    print("  best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # CSV 저장
    trials_df = study.trials_dataframe()
    csv_path = Path(cfg["paths"]["hpo_dir"]) / "trials.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(csv_path, index=False)
    print(f"\n  trials.csv → {csv_path}")

    # best_config.yaml
    out = export_best_config(cfg)
    print(f"  best_config.yaml → {out}")


if __name__ == "__main__":
    main()
