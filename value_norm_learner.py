"""Value-normalized PPO Torch Learner (MAPPO ValueNorm).

`PPOTorchLearner.compute_loss_for_module` 를 1:1 복제하되 **value-function 손실 블록만** 정규화
버전으로 치환한다 (surrogate·entropy·kl·로깅 키는 그대로). 크리틱(module.value_normalizer 보유)이
정규화 공간 z 를 출력(`compute_values`→denorm), 손실은 σ 로 나눠 *정규화 단위* 에서 계산한다:

    vf_loss = ((value_fn_out − value_targets) / σ)²      # μ 는 차분에서 상쇄

이로써 PPO 의 `vf_clip_param` clamp(제곱손실에 적용)가 **σ-단위 outlier guard** 로 바뀐다
(스케일 독립). return 스케일이 비정상이라 고정 clip 이 옳을 수 없던 value 학습 붕괴(saga)를 끝낸다.
상세·근거: CLAUDE.md "Value-function 학습 붕괴" / value_norm.py.

⚠️ ray 2.55.1 커플링: 아래 메서드는 `ray.rllib.algorithms.ppo.torch.ppo_torch_learner.
PPOTorchLearner.compute_loss_for_module` (2.55.1) 의 복제다. RLlib 업그레이드 시 parent 원본과
diff 를 확인해 vf 블록 외 변경분을 반영할 것.
"""
from typing import Any, Dict

from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
    PPOConfig,
)
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import ENTROPY_KEY, POLICY_LOSS_KEY, VF_LOSS_KEY
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.utils.typing import ModuleID, TensorType

torch, nn = try_import_torch()


class ValueNormPPOTorchLearner(PPOTorchLearner):
    """PPOTorchLearner with value-target normalization in the VF loss."""

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        module = self.module[module_id].unwrapped()

        if Columns.LOSS_MASK in batch:
            mask = batch[Columns.LOSS_MASK]
            num_valid = torch.sum(mask)

            def possibly_masked_mean(data_):
                return torch.sum(data_[mask]) / num_valid

        else:
            mask = None
            possibly_masked_mean = torch.mean

        action_dist_class_train = module.get_train_action_dist_cls()
        action_dist_class_exploration = module.get_exploration_action_dist_cls()

        curr_action_dist = action_dist_class_train.from_logits(
            fwd_out[Columns.ACTION_DIST_INPUTS]
        )
        prev_action_dist = action_dist_class_exploration.from_logits(
            batch[Columns.ACTION_DIST_INPUTS]
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(batch[Columns.ACTIONS]) - batch[Columns.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if config.use_kl_loss:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = possibly_masked_mean(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = possibly_masked_mean(curr_entropy)

        surrogate_loss = torch.min(
            batch[Postprocessing.ADVANTAGES] * logp_ratio,
            batch[Postprocessing.ADVANTAGES]
            * torch.clamp(logp_ratio, 1 - config.clip_param, 1 + config.clip_param),
        )

        # ── Value function loss (value-target normalized) ─────────────────────
        if config.use_critic:
            value_targets = batch[Postprocessing.VALUE_TARGETS]
            value_normalizer = getattr(module, "value_normalizer", None)

            # 정규화기 갱신을 compute_values 호출 *전* 에 수행 → denorm 이 갱신된 stats 사용 →
            # z 복원이 일관 (μ,σ 변동분이 손실/예측에 동일 반영). update 는 no_grad.
            if value_normalizer is not None:
                with torch.no_grad():
                    value_normalizer.update(
                        value_targets if mask is None else value_targets[mask]
                    )

            value_fn_out = module.compute_values(
                batch, embeddings=fwd_out.get(Columns.EMBEDDINGS)
            )

            if value_normalizer is not None:
                _, std = value_normalizer.stats()
                # 정규화 단위 손실: ((v − G)/σ)². μ 는 차분에서 상쇄(=z − normalize(G)).
                vf_loss = torch.pow((value_fn_out - value_targets) / std, 2.0)
            else:
                # 정규화기 없음(legacy 모듈) → 표준 제곱손실 (방어적 fallback).
                vf_loss = torch.pow(value_fn_out - value_targets, 2.0)

            vf_loss_clipped = torch.clamp(vf_loss, 0, config.vf_clip_param)
            mean_vf_loss = possibly_masked_mean(vf_loss_clipped)
            mean_vf_unclipped_loss = possibly_masked_mean(vf_loss)
        else:
            z = torch.tensor(0.0, device=surrogate_loss.device)
            value_fn_out = mean_vf_unclipped_loss = vf_loss_clipped = mean_vf_loss = z

        total_loss = possibly_masked_mean(
            -surrogate_loss
            + config.vf_loss_coeff * vf_loss_clipped
            - (
                self.entropy_coeff_schedulers_per_module[module_id].get_current_value()
                * curr_entropy
            )
        )

        if config.use_kl_loss:
            total_loss += self.curr_kl_coeffs_per_module[module_id] * mean_kl_loss

        # explained_var 는 real-space (value_fn_out=denorm, value_targets=real) → 기존과 비교가능.
        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: -possibly_masked_mean(surrogate_loss),
                VF_LOSS_KEY: mean_vf_loss,
                LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY: mean_vf_unclipped_loss,
                LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: explained_variance(
                    batch[Postprocessing.VALUE_TARGETS], value_fn_out
                ),
                ENTROPY_KEY: mean_entropy,
                LEARNER_RESULTS_KL_KEY: mean_kl_loss,
            },
            key=module_id,
            window=1,
        )
        return total_loss
