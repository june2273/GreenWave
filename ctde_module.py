"""
CTDE (Centralized Training, Decentralized Execution) RLModule for MAPPO.

Actor sees only its own local observation; Critic sees the concatenated
global observation (all agents). At execution time the critic is unused,
so each agent only needs its own local obs.

Expected obs space (per agent, Dict):
    {
      "local":  Box(shape=(D,)),
      "global": Box(shape=(D * N_agents,)),
    }

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
    """PPO RLModule with separate actor (local obs) / critic (global obs) encoders.

    Bypasses PPOCatalog (which assumes a single shared encoder) and constructs
    two independent MLP encoders + heads in `setup()`.
    """

    @override(RLModule)
    def setup(self):
        local_space = self.observation_space["local"]
        global_space = self.observation_space["global"]
        local_dim = int(local_space.shape[0])
        global_dim = int(global_space.shape[0])
        n_actions = int(self.action_space.n)
        h = int(self.model_config.get("hidden_dim", 128))

        self.pi_encoder = nn.Sequential(
            nn.Linear(local_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
        )
        self.vf_encoder = nn.Sequential(
            nn.Linear(global_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
        )
        self.pi = nn.Linear(h, n_actions)
        self.vf = nn.Linear(h, 1)

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
        # Local-only actor — used for inference and exploration.
        local = batch[Columns.OBS]["local"]
        logits = self.pi(self.pi_encoder(local))
        return {Columns.ACTION_DIST_INPUTS: logits}

    @override(RLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Actor still uses only local obs. We deliberately do NOT emit
        # Columns.EMBEDDINGS — this forces the PPO learner to invoke
        # compute_values(batch, embeddings=None), where our override
        # re-encodes from the global observation.
        return self._forward(batch, **kwargs)

    @override(DefaultPPOTorchRLModule)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        # Critic always uses the global observation (centralized critic).
        global_obs = batch[Columns.OBS]["global"]
        return self.vf(self.vf_encoder(global_obs)).squeeze(-1)

    @override(InferenceOnlyAPI)
    def get_non_inference_attributes(self) -> List[str]:
        # Strip critic-side parameters on inference-only EnvRunner workers.
        # Do NOT call super() — the default implementation references
        # `encoder.critic_encoder`, which we do not have.
        return ["vf", "vf_encoder"]
