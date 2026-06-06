"""
CTDE (Centralized Training, Decentralized Execution) RLModule for MAPPO.

Actor sees only its own local observation; Critic is **agent-aware** — it sees
the FULL obs (own local D | global D*N). The leading D dims (this agent's own
local obs) act as an agent identifier so the critic can produce per-agent values
(agent-specific global state, MAPPO Yu 2021), avoiding the "identical global
input / different per-agent target" ambiguity of a global-only critic. At
execution time the critic is unused, so each agent only needs its own local obs.

Expected obs space (per agent, **single flat Box**):
    Box(shape=(D + D*N_agents,))
    - First D dims = this agent's local obs
    - Remaining D*N dims = all agents' local obs concatenated (in possible_agents order)

The module slices the Box internally so the actor only sees the first D dims;
the critic consumes the whole Box.
Dict obs (`{"local", "global"}`) was tried first but RLlib worker processes
(num_env_runners ≥ 1) silently drop Dict subspaces through the connector
pipeline → total_steps=0. Flat Box is robust across single/multi-worker.

`local_dim` (= D) must be passed via `model_config["local_dim"]` since the
module cannot infer it from the flat Box shape alone.

Action space: Discrete(num_green).
"""
from typing import Any, Dict, List, Optional

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.distribution.torch.torch_distribution import TorchCategorical
from ray.rllib.core.rl_module.apis.inference_only_api import InferenceOnlyAPI
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()


class CentralizedCriticPPOModule(DefaultPPOTorchRLModule):
    """PPO RLModule with separate actor (local obs) / critic (full agent-aware obs) encoders.

    Bypasses PPOCatalog (which assumes a single shared encoder) and constructs
    two independent MLP encoders + heads in `setup()`.
    """

    @override(RLModule)
    def setup(self):
        # Flat Box obs: [local (D) | global (D*N)]. local_dim must come from
        # model_config since we can't infer D from total dim alone (don't know N).
        total_dim = int(self.observation_space.shape[0])
        local_dim = int(self.model_config.get("local_dim", 0))
        if local_dim <= 0 or local_dim > total_dim:
            raise ValueError(
                f"CentralizedCriticPPOModule requires model_config['local_dim'] "
                f"in (0, {total_dim}]; got {local_dim}. "
                f"Set it to env._obs_dim when building the RLModuleSpec."
            )
        self._local_dim = local_dim
        n_actions = int(self.action_space.n)
        h = int(self.model_config.get("hidden_dim", 128))
        # Critic 폭은 actor 와 분리 (학습 전용 → inference 비용 0). N=6 에서 150-d
        # global concat 을 fit 하려면 128 로는 부족(입력>hidden 압축병목) → 256.
        h_vf = int(self.model_config.get("vf_hidden_dim", 256))

        # Actor: local obs (D) 만. Decentralized 실행 보존.
        self.pi_encoder = nn.Sequential(
            nn.Linear(local_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
        )
        # Critic: 전체 obs (local D | global D*N) = total_dim 을 입력으로.
        # 앞 D 차원(자기 local)이 agent 식별자 역할 → per-agent value 학습 가능
        # (global-only 일 때의 "동일입력/다른타깃" ambiguity 제거, agent-specific
        # global state, MAPPO Yu 2021). LayerNorm 으로 150-d concat 조건수 개선.
        self.vf_encoder = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, h_vf), nn.Tanh(),
            nn.Linear(h_vf, h_vf), nn.Tanh(),
        )
        self.pi = nn.Linear(h, n_actions)
        self.vf = nn.Linear(h_vf, 1)
        # Value head 작은 init → 초기 value 폭주 억제, VF 빠른 수렴 (orthogonal, MAPPO).
        nn.init.orthogonal_(self.vf.weight, gain=0.01)
        nn.init.zeros_(self.vf.bias)

        self.action_dist_cls = TorchCategorical

    @override(RLModule)
    def get_inference_action_dist_cls(self):
        return TorchCategorical

    @override(RLModule)
    def get_exploration_action_dist_cls(self):
        return TorchCategorical

    @override(RLModule)
    def get_train_action_dist_cls(self):
        return TorchCategorical

    @override(RLModule)
    def get_initial_state(self) -> dict:
        return {}

    @override(RLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Local-only actor — slice first local_dim cols from flat obs.
        # Decentralized execution: actor never reads beyond [:, :local_dim].
        local = batch[Columns.OBS][..., : self._local_dim]
        logits = self.pi(self.pi_encoder(local))
        return {Columns.ACTION_DIST_INPUTS: logits}

    @override(RLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Actor still uses only local obs. We deliberately do NOT emit
        # Columns.EMBEDDINGS — this forces the PPO learner to invoke
        # compute_values(batch, embeddings=None), where our override
        # re-encodes from the global slice.
        return self._forward(batch, **kwargs)

    @override(DefaultPPOTorchRLModule)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        # Critic uses the FULL obs (own local D | global D*N). 앞 D 차원이 agent
        # 식별자 역할 → per-agent value 학습 가능 (agent-aware centralized critic).
        full_obs = batch[Columns.OBS]
        return self.vf(self.vf_encoder(full_obs)).squeeze(-1)

    @override(InferenceOnlyAPI)
    def get_non_inference_attributes(self) -> List[str]:
        # Strip critic-side parameters on inference-only EnvRunner workers.
        # Do NOT call super() — the default implementation references
        # `encoder.critic_encoder`, which we do not have.
        return ["vf", "vf_encoder"]
