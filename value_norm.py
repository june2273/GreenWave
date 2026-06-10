"""Value-target normalizer (MAPPO ValueNorm; van Hasselt 2016 PopArt 의 Adaptive-Rescaling 부분).

크리틱이 **정규화 공간** 값 `z` 를 출력하고, `denormalize(z)=σz+μ` 로 real-space 값을 복원한다
(GAE·inference·explained_var 는 real-space 를 사용). 손실은 정규화 공간에서 `normalize(target)` 과
비교한다. 이로써:
- value 타깃이 O(1) → PPO 의 `vf_clip_param` clamp 가 *제곱손실에* 걸려도 거의 안 잘림(=신호 보존)
  AND clip 임계가 **σ-단위(스케일 독립)** 가 됨. return 스케일이 비정상(정책·regime 따라 변동)이라
  고정 `vf_clip` 이 옳을 수 없던 문제(=value 학습 붕괴 saga)를 끝냄. 상세: CLAUDE.md "Value-function 학습 붕괴".
- 크리틱 출력에 `+μ` 구조적 offset → init z≈0 이어도 출력≈μ → cold-start 에 모든 샘플이 clip 밖으로
  나가 gradient 가 0 으로 얼어붙는 freeze 를 구조적으로 방지(특히 dense warmup 의 저분산 regime).

EMA + debiasing 으로 **첫 update 부터 unbiased** (cold start warmup 불요). scalar value 1개에 대한
running stat 을 버퍼로 보유 → RLModule(nn.Module)의 state_dict 에 포함되어 체크포인트에 저장된다.
"""
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

# 정규화기 하이퍼파라미터 (CLAUDE.md "Value-function 학습 붕괴" 참조).
#   _VN_BETA   : running stat EMA decay. **MAPPO 표준 0.99999** — 채택 근거는 *MAPPO 원
#                구현과의 정렬*뿐. "느린 β → σ 안정 → explained_var 노이즈↓" 가설은 0.999 vs
#                0.99999 A/B 로 **반증됨**(노이즈 차이 없음; 노이즈 주범 = window=1 로깅 +
#                미수렴 정책). 무해해서 표준값 유지. ⚠️ 느린 EMA 는 regime shift(예: warmup→
#                dense resume, return 스케일 수 배 점프)에서 σ 가 못 따라가 σ-단위 clip 이
#                재동결될 수 있음 — dense 진입 직후 vf_loss_unclipped vs vf_loss 갭 확인,
#                재발 시 normalizer 버퍼 리셋 또는 --vn-beta 하향(예 0.9999).
#                "더 느린 β = 더 안정" 으로 일반화하지 말 것.
#   _VAR_FLOOR : variance 하한. cold-start 0除算 방지(steady state var≫이라 무관).
#   _DEBIAS_EPS: debiasing term 하한.
_VN_BETA = 0.99999
_VAR_FLOOR = 1e-2
_DEBIAS_EPS = 1e-5


class ValueNorm(nn.Module):
    """Scalar running mean/std normalizer with EMA + debiasing.

    버퍼 3개(`running_mean`, `running_mean_sq`, `debias`)로 1-d value 분포를 추적한다.
    `update()` 는 학습 손실에서 `no_grad` 로 value 타깃 배치마다 호출.
    """

    def __init__(self, beta: float = _VN_BETA, var_floor: float = _VAR_FLOOR):
        super().__init__()
        self.beta = float(beta)
        self.var_floor = float(var_floor)
        # scalar buffers (state_dict 에 저장 → 체크포인트 보존).
        # float64 인 이유: var = E[x²]−E[x]² 의 캔슬레이션 — |μ|≈4e3 이면 mean_sq≈1.6e7
        # 로 float32 정밀도 한계(2²⁴)에 닿아 σ 가 왜곡될 수 있다. 구 체크포인트의
        # float32 버퍼 로드는 load_state_dict 의 copy_ 캐스트로 자동 호환.
        self.register_buffer("running_mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("running_mean_sq", torch.zeros((), dtype=torch.float64))
        self.register_buffer("debias", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: "torch.Tensor") -> None:
        """value 타깃 배치로 running mean/mean_sq/debias 를 EMA 갱신."""
        x = x.detach().to(torch.float64)
        batch_mean = x.mean()
        batch_mean_sq = (x * x).mean()
        w = self.beta
        self.running_mean.mul_(w).add_(batch_mean * (1.0 - w))
        self.running_mean_sq.mul_(w).add_(batch_mean_sq * (1.0 - w))
        self.debias.mul_(w).add_(1.0 - w)

    def stats(self):
        """debias 보정된 (mean, std) 반환. update 전엔 (0, √var_floor)."""
        d = self.debias.clamp(min=_DEBIAS_EPS)
        mean = self.running_mean / d
        mean_sq = self.running_mean_sq / d
        var = (mean_sq - mean * mean).clamp(min=self.var_floor)
        return mean, var.sqrt()

    def normalize(self, x: "torch.Tensor") -> "torch.Tensor":
        """real-space → 정규화 공간 `(x − μ)/σ`. (입력 dtype 보존 — float64 stat 와 혼합 방지)"""
        mean, std = self.stats()
        return ((x.to(torch.float64) - mean) / std).to(x.dtype)

    def denormalize(self, z: "torch.Tensor") -> "torch.Tensor":
        """정규화 공간 → real-space `z·σ + μ`. (크리틱 출력 → GAE/inference 용 value;
        입력 dtype 보존)"""
        mean, std = self.stats()
        return (z.to(torch.float64) * std + mean).to(z.dtype)
