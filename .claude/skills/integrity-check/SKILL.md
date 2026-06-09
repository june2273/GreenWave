---
name: integrity-check
description: GreenWave 레포에서 코드 변경 후 파이프라인(train→eval→record→stats) 무결성을 검사하고, 에러를 수정하고, 변경 내용을 md 문서에 반영하고, 레포 컨벤션에 맞는 커밋명을 제안할 때 사용. 트리거 예시 "무결성 검사", "파이프라인 점검", "커밋 전 검사", "integrity check", "변경 검증하고 문서 반영하고 커밋명 알려줘".
version: 0.1.0
---

# GreenWave Integrity Check

코드를 수정한 뒤(특히 CLI 인자·env 파라미터·reward/obs 변경 후) 파이프라인 전체가
일관·동작하는지 검증하고, md 문서를 동기화하고, 커밋명을 제안하는 절차.

핵심 원칙: **"0회 등장 = 삭제 가능"이 아니다.** env가 구조적으로 요구하는 값이거나
CLAUDE.md가 재현용으로 보존하라고 명시한 것은 빼지 않는다. 삭제 전 항상 결합도와
문서 근거를 확인한다.

## 0. 사전 — 환경 변수

SUMO 의존 검사에는 항상 먼저:
```bash
export SUMO_HOME=$(brew --prefix sumo)/share/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```
파이썬은 `.venv/bin/python` 사용 (시스템 `python` 없음).

## 1. 변경 범위 파악

```bash
git status --short && git diff --stat
git log --oneline -6          # 커밋 컨벤션·이미 커밋된 기능 확인
```
- working tree에 **이번 작업 외 미커밋 변경**이 섞여있는지 식별 (커밋 범위 판단용).
- 변경된 기능이 **이미 HEAD에 커밋됐는지** 확인: `git show HEAD:<file> | grep -c <symbol>`.

## 2. 파이프라인 일관성 — 메타데이터 흐름 추적

obs/reward 차원에 영향을 주는 필드는 **train→eval→record→env 4곳**에 일관 배선돼야 한다.
대표 필드: `neighbor_obs upstream_phase progression_coeff brt_prog_weight ctde_mode reward_mode time_to_teleport brt_weight`.

```bash
for f in train_mappo.py evaluate_mappo.py record_video_mappo.py record_video_fixed.py env_sumo_pz.py ctde_module.py; do
  echo "===== $f ====="; grep -n "<필드들 | 로 구분>" "$f"; done
```
체크리스트:
- train: `env_config` dict + `train_metadata.json` 양쪽에 기록되는가.
- eval/record_mappo: 모델별 `train_metadata.json`에서 **개별 로드**되는가 (옛/새 모델 혼재 대비).
- obs 차원 바꾸는 플래그(neighbor_obs/upstream_phase)는 resume guard가 **거부**, reward-only(progression)는 **경고만**.
- record_mappo는 reward-only 필드(progression)를 **추론에 안 넘기는 게 정상**.

## 3. 제거/변경 시 하류 의존성 확인

인자·파라미터를 지웠다면:
```bash
# leftover 참조 (코드 0개여야 함)
grep -rn "<지운심볼1>\|<지운심볼2>" *.py
# md 문서 stale 참조 (전부 clean이어야 함)
grep -rn "<지운플래그>" --include="*.md" .
# CSV/후처리 의존 (stats_eval가 지운 컬럼 읽나?)
grep -n "<지운컬럼>" stats_eval.py
```
- env 파라미터를 지웠으면 **모든 호출부**(eval의 `SumoParallelEnv(...)` kwargs 등)도 같이 수정.
- 보조 함수/dict가 분기 삭제 후에도 여전히 정의·사용되는지 확인 (`_phase_from_secs`, `SEJONG_PER_TLS_PHASE_SECONDS` 등).

## 4. 런타임 스모크 (가장 가치 높음)

green-wave 풀스택 + legacy off를 실제 구성해 차원·스텝 검증:
```bash
.venv/bin/python - <<'PY'
import numpy as np
from env_sumo_pz import SumoParallelEnv
from map_presets import resolve_map_args
cfg, tls = resolve_map_args(map_name="3x2-brt", sumo_cfg_arg=None, tls_ids_arg=None, traffic="high")
env = SumoParallelEnv(sumo_cfg=cfg, tls_ids=tls, use_gui=False, ctde_mode=True,
                      neighbor_obs=True, upstream_phase=True, progression_coeff=1.0, brt_prog_weight=3.0)
assert sorted(env._corridor_agents) == ["tl_0","tl_2","tl_4"]      # 회랑 자동식별
assert env._obs_dim == 39                                          # actor enriched
a0 = env.possible_agents[0]
assert env.observation_space(a0).shape == (201,)                   # CTDE critic
obs,_ = env.reset(); assert obs[a0].shape == (201,)
for _ in range(20):
    obs,r,t,tr,info = env.step({ag: np.random.randint(0, env.action_space(ag).n) for ag in env.agents})
    if not env.agents: break
k = list(info)[0]; assert "corridor_brt_speed_ratio" in info[k]
env.close()
env2 = SumoParallelEnv(sumo_cfg=cfg, tls_ids=tls, use_gui=False)
assert env2._obs_dim == 25; env2.close()                           # legacy 회귀 없음
print("SMOKE OK")
PY
```
**기대 기준치** (3x2-brt): corridor `{tl_0,tl_2,tl_4}` · obs `25`(off)→`29`(neighbor)→`39`(+upstream) · critic `151`→`201` · neighbor_dir[tl_2]=`{N:tl_0,E:tl_3,S:tl_4}`.
(스모크 중 SUMO `Retrying in 1 seconds` 로그는 정상.)

## 5. argparse + 컴파일

인자를 건드린 스크립트는 `--help`로 top-level import·argparse 무결성 확인:
```bash
.venv/bin/python evaluate_mappo.py --help    >/dev/null && echo eval OK
.venv/bin/python record_video_fixed.py --help >/dev/null && echo fixed OK
.venv/bin/python train_mappo.py --help       >/dev/null && echo train OK
.venv/bin/python -m py_compile train_mappo.py evaluate_mappo.py record_video_mappo.py record_video_fixed.py env_sumo_pz.py stats_eval.py ctde_module.py && echo "compile OK"
```

## 6. md 문서 반영

변경을 다음 문서에 동기화 (해당하는 곳만):
- **CLAUDE.md**: Commands 예시, 플래그 설명, "미사용/더미 코드 인벤토리", PPO 표, Key Constraints.
- **README.md**: 사용자용 명령·데이터 출처.
- **DESIGN_progression_greenwave.md**: green-wave 설계 수치.

원칙: 동작이 안 바뀌면(예: 균등 사이클 15s/phase 유지) 서술 추가 불필요 — stale 참조 제거에 집중.
최종 sweep: `grep -rn "<지운것들>" --include="*.md" .` 가 clean이어야 종료.

## 7. 커밋명 제안 (커밋은 사용자 요청 시에만)

레포 컨벤션 = `X.Y.Z 한국어 요약` (예: `0.9.1 greenwave를 위한 보상함수 무정차 통과 항 추가`).
`git log --oneline`로 최신 버전 확인 후 다음 번호 제안. 본문에 변경 항목 bullet.
대안으로 conventional commits(`chore:`/`fix:`/`docs:`) 형식도 함께 제시.
**커밋·푸시는 사용자가 명시적으로 요청할 때만** 수행한다 (프로젝트 규칙).

## 보고 형식

검사 결과를 표로 (항목 | 결과), 발견된 에러는 수정 후 전후 명시, md 반영 내역,
커밋명 제안 순으로 간결히 보고한다.
