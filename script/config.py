"""
Config 로더.

사용 예:
    from config import load_config
    cfg = load_config()              # 기본 경로: script/config.yaml
    cfg = load_config("other.yaml")  # 다른 config 사용
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    YAML config 로드 + 상대 경로를 절대 경로로 변환.

    Returns:
        dict: 파싱된 config. `paths.*`는 절대 경로 문자열로 변환됨.
    """
    path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    base = path.parent
    cfg["paths"] = {
        k: str((base / v).resolve()) for k, v in cfg["paths"].items()
    }

    # Optuna storage의 상대 경로(sqlite:///../...)도 config 파일 기준으로 변환
    storage = cfg.get("hpo", {}).get("storage", "")
    if storage.startswith("sqlite:///") and not storage.startswith("sqlite:////"):
        rel = storage[len("sqlite:///"):]
        abs_path = (base / rel).resolve()
        cfg["hpo"]["storage"] = f"sqlite:///{abs_path}"

    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    """필수 키 존재 여부만 최소 검증. 타입 검증은 사용부에서."""
    required_top = ["paths", "data", "rolling_window", "env", "ppo", "train"]
    for k in required_top:
        if k not in cfg:
            raise KeyError(f"config 누락: '{k}' 섹션이 필요합니다.")

    # Transformer 제약: d_model은 nhead의 배수여야 함
    if "transformer" in cfg:
        t = cfg["transformer"]
        if t["d_model"] % t["nhead"] != 0:
            raise ValueError(
                f"transformer.d_model({t['d_model']})은 nhead({t['nhead']})의 "
                f"배수여야 합니다."
            )


def parse_net_arch(value) -> list[int]:
    """
    Optuna categorical로 넘긴 '64,64' 같은 문자열을 [64, 64]로 변환.
    이미 리스트면 그대로 반환.
    """
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",")]
    return list(value)


if __name__ == "__main__":
    # 간단한 점검용
    cfg = load_config()
    print("=" * 60)
    print(" Config 로드 점검")
    print("=" * 60)
    print(f"data_csv   : {cfg['paths']['data_csv']}")
    print(f"models_dir : {cfg['paths']['models_dir']}")
    print(f"results_dir: {cfg['paths']['results_dir']}")
    print(f"window     : {cfg['data']['window']}")
    print(f"n_regimes  : {cfg['data']['n_regimes']}")
    print(f"n_assets   : {len(cfg['data']['asset_cols'])}")
    print(f"ppo.lr     : {cfg['ppo']['learning_rate']}")
    print(f"ppo.net_arch: {cfg['ppo']['net_arch']}")
    print(f"train.total_timesteps: {cfg['train']['total_timesteps']:,}")
    print(f"train.total_timesteps_smoke: {cfg['train']['total_timesteps_smoke']:,}")
    print(f"hpo.storage: {cfg['hpo']['storage']}")
