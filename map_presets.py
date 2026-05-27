"""
Map preset 정의 + 인자 해상 (train_mappo / evaluate_mappo / record_video_mappo 공용)

--map 옵션이 (sumo_cfg, default tls_ids) 를 결정.
--sumo-cfg / --tls-ids 가 명시되면 그것이 우선.

시나리오 역할 이분화:
  2x2 / 2x2-brt  → 모델 개발·빠른 실험용
  3x2-brt        → 세종시 현실 시뮬레이션 최종 버전 (--traffic high 필수)

--traffic high 분기:
  2x2      → 2x2grid_dense.sumocfg  (sumo-rl dense, legacy 호환)
  2x2-brt  → 2x2_brt_dense.sumocfg  (행복청 실측 ~6,810 veh/h, 2x2 기준)
  3x2-brt  → 3x2_brt_dense.sumocfg  (행복청 실측 6교차로 기반, 세종 최종)
"""
from pathlib import Path
from typing import List, Optional

MAP_CHOICES = ["single", "2x2", "2x2-brt", "3x2", "3x2-brt"]

# sumo_cfg 는 GreenWave/ 기준 상대경로; tls_ids 는 SUMO 네트워크의 TLS id 목록.
MAP_PRESETS = {
    "single":  {"sumo_cfg": None,                                "tls_ids": ["C"]},
    "2x2":     {"sumo_cfg": "sumo_data/2x2/2x2grid.sumocfg",     "tls_ids": ["1","2","5","6"]},
    "2x2-brt": {"sumo_cfg": "sumo_data/2x2_brt/2x2_brt.sumocfg", "tls_ids": ["1","2","5","6"]},
    "3x2":     {"sumo_cfg": "sumo_data/3x2/3x2.sumocfg",         "tls_ids": ["1","2","3","4","5","6"]},
    "3x2-brt": {"sumo_cfg": "sumo_data/3x2_brt/3x2_brt.sumocfg", "tls_ids": ["1","2","3","4","5","6"]},
}


def resolve_map_args(
    map_name: str,
    sumo_cfg_arg: Optional[str],
    tls_ids_arg: Optional[List[str]],
    traffic: str = "default",
    base_dir: Optional[Path] = None,
):
    """
    Returns
    -------
    (sumo_cfg_path: Optional[str], tls_ids: List[str])
        sumo_cfg_path: 절대경로 string (preset 사용 시) 또는 사용자 명시값.
                       single map + 명시 안 했으면 None (env가 기본 single.sumocfg 사용).
        tls_ids: 명시 안 했으면 preset 값.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    if map_name not in MAP_PRESETS:
        raise ValueError(f"Unknown map: {map_name!r}. Choices: {MAP_CHOICES}")

    preset = MAP_PRESETS[map_name]

    # tls_ids: 사용자 명시 우선
    tls_ids = list(tls_ids_arg) if tls_ids_arg else list(preset["tls_ids"])

    # sumo_cfg: 사용자 명시 > traffic=high 특수처리 > preset
    if sumo_cfg_arg:
        sumo_cfg = sumo_cfg_arg
    elif map_name == "2x2" and traffic == "high":
        sumo_cfg = "sumo_data/2x2/2x2grid_dense.sumocfg"
    elif map_name == "2x2-brt" and traffic == "high":
        sumo_cfg = "sumo_data/2x2_brt/2x2_brt_dense.sumocfg"
    elif map_name == "3x2-brt" and traffic == "high":
        sumo_cfg = "sumo_data/3x2_brt/3x2_brt_dense.sumocfg"
    else:
        sumo_cfg = preset["sumo_cfg"]

    # 상대경로면 GreenWave/ 루트 기준 절대경로화
    if sumo_cfg and not Path(sumo_cfg).is_absolute():
        sumo_cfg = str((base_dir / sumo_cfg).resolve())

    return sumo_cfg, tls_ids
