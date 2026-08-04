import torch


class WorldModelInference(torch.nn.Module):
    def __init__(self, world_model):
        super().__init__()
        self.encoder = world_model.encoder
        self.dynamics = world_model.dynamics
        # Dreamer's default initial state is learned, not all-zero.  obs_step()
        # resets it through Python control flow, which torch.onnx.export cannot
        # preserve when tracing a dummy is_first=False input.  Keep the exact
        # checkpoint initial state as buffers and apply the reset with tensor
        # operations in forward() so ONNX and PyTorch start from the same latent.
        initial = self.dynamics.initial(1)
        self.register_buffer("_initial_logit", initial["logit"].detach().clone())
        self.register_buffer("_initial_stoch", initial["stoch"].detach().clone())
        self.register_buffer("_initial_deter", initial["deter"].detach().clone())

    def forward(
        self,
        prop,
        image,
        prev_logit,
        prev_stoch,
        prev_deter,
        prev_action,
        is_first,
    ):
        reset = is_first.to(dtype=prev_deter.dtype).reshape(-1, 1)

        def reset_state(value, initial):
            mask = reset.reshape(
                reset.shape + (1,) * (value.ndim - reset.ndim)
            )
            initial = initial.expand_as(value)
            return value * (1.0 - mask) + initial * mask

        prev_state = {
            "logit": reset_state(prev_logit, self._initial_logit),
            "stoch": reset_state(prev_stoch, self._initial_stoch),
            "deter": reset_state(prev_deter, self._initial_deter),
        }
        prev_action = prev_action * (1.0 - reset)

        wm_obs = {
            "prop": prop,
            "image": image,
            "is_first": is_first,
        }

        embed = self.encoder(wm_obs)
        post, _ = self.dynamics.obs_step(
            prev_state,
            prev_action,
            embed,
            torch.zeros_like(is_first),
            sample=False,
        )

        wm_feature = self.dynamics.get_deter_feat(post)
        # prev_logit is part of the recurrent-state ABI even though Dreamer's
        # transition only consumes stoch+deter.  Keep it in the ONNX inputs.
        wm_feature = wm_feature + prev_logit.sum() * 0

        return (
            post["logit"],
            post["stoch"],
            post["deter"],
            wm_feature,
        )


class PolicyInference(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.history_encoder = actor_critic.history_encoder
        self.wm_feature_encoder = actor_critic.wm_feature_encoder
        self.actor = actor_critic.actor
        self.privileged_dim = actor_critic.privileged_dim

    def forward(self, command, history, wm_feature):
        hist_latent = self.history_encoder(history)
        wm_latent = self.wm_feature_encoder(wm_feature)
        x = torch.cat([hist_latent, command, wm_latent], dim=-1)
        return self.actor(x)
