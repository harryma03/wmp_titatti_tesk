# direct_recurrent_depth.py
# Deterministic recurrent-depth baseline for WL-WMP.
#
# Actor-visible state:
#   h_t in R^{dyn_deter} (default 512)
#
# Inputs at each perception update:
#   encoded depth + encoded prop + previous action history
#
# Training:
#   current depth/prop reconstruction with the same encoder/decoder settings,
#   optimizer, batch length, and update cadence as the RSSM.
#   No stochastic latent, prior/posterior, or KL loss.

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from . import networks
from . import tools


def _to_np(x: torch.Tensor):
    return x.detach().cpu().numpy()


class DirectRecurrentDepthModel(nn.Module):
    """Auditable deterministic recurrent-depth baseline.

    This class intentionally exposes the same minimum interface used by
    WMPRunner:
      - encoder(obs)
      - obs_step(prev_state, action, embed, is_first)
      - get_deter_feat(state)
      - _train(batch)
      - _model_opt
      - device
    """

    def __init__(self, config, obs_shape: Dict[str, tuple], use_camera: bool):
        super().__init__()
        self._config = config
        self.device = config.device
        self._use_amp = config.precision == 16
        self.hidden_dim = int(config.dyn_deter)
        self.action_dim = int(config.num_actions)

        # Exactly reuse the RSSM observation encoder.
        self.encoder = networks.MultiEncoder(
            obs_shape, **config.encoder, use_camera=use_camera
        )
        self.embed_size = int(self.encoder.outdim)

        act_cls = getattr(nn, config.act)
        input_layers = [
            nn.Linear(self.embed_size + self.action_dim,
                      int(config.dyn_hidden), bias=False)
        ]
        if config.norm:
            input_layers.append(nn.LayerNorm(int(config.dyn_hidden), eps=1e-3))
        input_layers.append(act_cls())
        self.input_proj = nn.Sequential(*input_layers)
        self.input_proj.apply(tools.weight_init)

        # Reuse the same normalized GRU cell type and 512-D hidden size.
        self.cell = networks.GRUCell(
            int(config.dyn_hidden), self.hidden_dim, norm=config.norm
        )
        self.cell.apply(tools.weight_init)

        # Match the RSSM decoder-head input width (1024 categorical stochastic
        # features + 512 deterministic features = 1536 by default), while the
        # actor still receives only the 512-D recurrent hidden state.
        if config.dyn_discrete:
            self.head_feat_dim = (
                int(config.dyn_stoch) * int(config.dyn_discrete)
                + int(config.dyn_deter)
            )
        else:
            self.head_feat_dim = int(config.dyn_stoch) + int(config.dyn_deter)

        self.head_readout = nn.Linear(self.hidden_dim, self.head_feat_dim)
        self.head_readout.apply(tools.weight_init)

        self.heads = nn.ModuleDict()
        self.heads["decoder"] = networks.MultiDecoder(
            self.head_feat_dim,
            obs_shape,
            **config.decoder,
            use_camera=use_camera,
        )
        # Keep the same zero-weight reward head for parameter/protocol parity.
        self.heads["reward"] = networks.MLP(
            self.head_feat_dim,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )

        self._scales = {
            "image": 1.0,
            "prop": 1.0,
            "reward": float(config.reward_head["loss_scale"]),
        }

        self._model_opt = tools.Optimizer(
            "direct_recurrent_depth",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"DirectRecurrentDepth parameters: {total:,}")
        print(
            "DirectRecurrentDepth contract: "
            f"embed={self.embed_size}, action={self.action_dim}, "
            f"hidden={self.hidden_dim}, head_feat={self.head_feat_dim}"
        )

    def initial(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.hidden_dim, device=torch.device(self.device)
        )

    def obs_step(
        self,
        prev_state: Optional[torch.Tensor],
        action: Optional[torch.Tensor],
        embed: torch.Tensor,
        is_first: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = embed.shape[0]
        if prev_state is None:
            prev_state = self.initial(batch_size)
        if action is None:
            action = torch.zeros(
                batch_size, self.action_dim, device=embed.device, dtype=embed.dtype
            )

        prev_state = prev_state.to(embed.device)
        action = action.to(embed.device)
        reset = is_first.to(embed.device, dtype=embed.dtype).reshape(batch_size, 1)
        prev_state = prev_state * (1.0 - reset)
        action = action * (1.0 - reset)

        x = self.input_proj(torch.cat([embed, action], dim=-1))
        _, state_list = self.cell(x, [prev_state])
        return state_list[0]

    def observe(
        self,
        embed: torch.Tensor,
        action: torch.Tensor,
        is_first: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # embed: [B,T,E], action: [B,T,A], is_first: [B,T]
        states = []
        h = state
        for t in range(embed.shape[1]):
            h = self.obs_step(h, action[:, t], embed[:, t], is_first[:, t])
            states.append(h)
        return torch.stack(states, dim=1)

    @staticmethod
    def get_deter_feat(state: torch.Tensor) -> torch.Tensor:
        return state

    def _train(self, data) -> Tuple[torch.Tensor, dict, dict]:
        data = self.preprocess(data)
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                states = self.observe(embed, data["action"], data["is_first"])
                head_feat = self.head_readout(states)

                predictions = {}
                for name, head in self.heads.items():
                    pred = head(head_feat)
                    if isinstance(pred, dict):
                        predictions.update(pred)
                    else:
                        predictions[name] = pred

                losses = {}
                for name, pred in predictions.items():
                    if name not in data:
                        continue
                    loss = -pred.log_prob(data[name])
                    if loss.shape != embed.shape[:2]:
                        raise RuntimeError(
                            f"{name} loss shape {loss.shape} != {embed.shape[:2]}"
                        )
                    losses[name] = loss

                if not losses:
                    raise RuntimeError("Direct recurrent baseline produced no losses.")

                scaled = {
                    name: value * self._scales.get(name, 1.0)
                    for name, value in losses.items()
                }
                model_loss = sum(scaled.values())

            metrics = self._model_opt(
                torch.mean(model_loss), self.parameters()
            )

        metrics.update({
            f"{name}_loss": _to_np(value)
            for name, value in losses.items()
        })
        metrics["model_loss"] = _to_np(torch.mean(model_loss))
        metrics["hidden_rms"] = _to_np(
            torch.sqrt(torch.mean(states.detach() ** 2))
        )

        context = {
            "embed": embed.detach(),
            "feat": states.detach(),
        }
        return states.detach(), context, metrics

    def preprocess(self, obs):
        if "is_first" not in obs:
            raise KeyError("is_first is required.")
        return {
            key: torch.as_tensor(value, device=self.device)
            for key, value in obs.items()
        }