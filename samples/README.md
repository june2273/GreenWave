# samples/ — 심사용 큐레이션 산출물

`models/`·`results/`·`videos/` 는 gitignore 라 재현 불가능한 휘발 산출물이고, 발표·심사에
쓸 확정본만 이 폴더에 보존한다. **메인 주장 자료는 `sejong 3-way/`** (세종시 현실 시나리오
최종 버전, 3x2-brt `--traffic high`, 수정 지표 2026-06-11 재평가본).

## 심사위원에게 보여줄 순서 (권장)

1. **`videos/sejong 3-way/3x2compare_3way_1080p.mp4`** — 세종 6교차로 LOS D 에서
   실측 Fixed-Time vs MAPPO vs CTDE 나란히 비교 영상 (시각적 핵심).
2. **`results/sejong 3-way/stats_summary_regen.txt`** — 발표용 헤드라인 자동 생성문:
   - MAPPO: 평균대기 **IQM 49.4% 개선** [95% CI 48.7~50.2], 30/30 에피소드 우월,
     Wilcoxon+Holm **p=5.59e-09, A12=1.00**
   - CTDE: 45.8% 개선 [45.0~46.7], 동일 유의수준
   - throughput +43% (5,772 vs 4,030 veh), travel −30%, CO₂ −35%
3. **`results/sejong 3-way/figs/`** — 헤드라인 그림 5장
   (대기시간·통과량 box/forest, 통행시간 forest; IQM±부트스트랩 95% CI).
4. **`results/sejong 3-way/eval_sejong_dense_30_regen.csv`** — 30-ep paired 원자료
   (+ `stats_tests_regen.csv` 검정 전체, `stats_ci_regen.csv` CI 전체).
5. (재현성 질문 대비) `models/sejong 3-way/` 체크포인트 +
   `results/sejong 3-way/{MAPPO,CTDE}_sejong*/` TensorBoard 이벤트 로그 +
   `results/sejong 3-way/TensorBoard.png` 학습 곡선.

## 평가 조건 (공정성)

- Fixed-Time 은 균등 사이클이 아니라 **세종시 실측 신호** (공공데이터포털 2023 +
  세담터 2026.04, 6 TLS 각각 다른 cycle — `SEJONG_PER_TLS_PHASE_SECONDS`, sim-time 기반).
- 세 알고리즘 동일 seed·동일 SUMO 설정의 paired 30-ep, `teleported≈0~24` 수준으로
  텔레포트 인공물 아님 검증, 통계는 Friedman→Wilcoxon+Holm→A12→부트스트랩 CI
  (`stats_eval.py`, Agarwal 2021 IQM 방식).
- ⚠️ 2026-06-11 이전 생성 CSV(`3-way/`, `sejong/` 폴더의 구 자료)는 waiting-time-memory
  포화 영향권 — 절대값 비교 금지, 참고용으로만 보존.

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `results/sejong 3-way/` | **메인**: 30-ep regen CSV + 통계(검정/CI/요약) + figs + TB 로그 |
| `videos/sejong 3-way/` | 3-way 비교 영상(1080p 합성 + 개별 cmp_*) |
| `models/sejong 3-way/` | MAPPO_sejong_dense · CTDE_sejong_dense 체크포인트 |
| `results/3-way/`, `videos/3-way/` | 개발용 2x2-brt 3-way (보조 — 모델 개발 과정 증빙) |
| `results/sejong/`, `videos/sejong/` | 초기 세종 비교 (구 지표, 참고용) |

## CTDE 위상 (정직 포지셔닝)

CTDE 는 Fixed 대비 45.8% 개선으로 유효하나 **MAPPO 대비 열위** (대기 97.7 vs 90.5,
p=5.6e-09; 분산도 3.6× 큼). 크리틱 정상화(value_norm, explained_var 0.076→0.5~0.7)
후 재학습도 헤드라인 회복에 실패 — CTDE 구조 개선은 본 프로젝트 범위에서 분리해
독립 후속 연구로 이관. 심사 자료에서는 "MAPPO 가 최종 제안, CTDE 는 한계 분석·향후
연구"로 서술.
