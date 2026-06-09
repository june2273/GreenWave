"""MAPPO(비-CTDE) 경로용 value-normalized PPO RLModule.

`DefaultPPOTorchRLModule` 을 얇게 확장해 크리틱이 **정규화 공간** value 를 출력하게 한다
(`compute_values` 가 `denorm(z)=σz+μ` 반환). CTDE 경로는 `ctde_module.CentralizedCriticPPOModule`
이 동일 처리를 자체 보유하므로 이 클래스는 비-CTDE(plain MAPPO) 경로에서만 주입된다.

`--value-norm` on 일 때만 train_mappo.py 가 이 모듈을 rl_module_spec 으로 주입(+ model_config
`value_norm=True`). off 면 RLlib 기본 모듈이 그대로 쓰여 legacy 거동 유지.
설계 근거: CLAUDE.md "Value-function 학습 붕괴" / value_norm.py 참조.
"""
from typing import Any, Dict, List, Optional

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.rl_module.apis.inference_only_api import InferenceOnlyAPI
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

from value_norm import ValueNorm, _VN_BETA


class NormalizedDefaultPPOTorchRLModule(DefaultPPOTorchRLModule):
    """DefaultPPO + value-target normalization (크리틱 정규화 출력)."""

    @override(RLModule)
    def setup(self):
        super().setup()
        self._value_norm = bool(self.model_config.get("value_norm", False))
        if self._value_norm:
            self.value_normalizer = ValueNorm(
                beta=float(self.model_config.get("vn_beta", _VN_BETA))
            )

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        # 기본 크리틱(catalog vf 인코더+헤드)의 raw 출력 = 정규화 공간 z.
        raw = super().compute_values(batch, embeddings)
        if getattr(self, "_value_norm", False):
            return self.value_normalizer.denormalize(raw)
        return raw

    @override(InferenceOnlyAPI)
    def get_non_inference_attributes(self) -> List[str]:
        attrs = super().get_non_inference_attributes()
        if getattr(self, "_value_norm", False):
            attrs.append("value_normalizer")
        return attrs
