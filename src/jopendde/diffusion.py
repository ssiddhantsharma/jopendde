"""DiffusionConditioning / DiffusionModule and the diffusion sampling loop.

FoldCP paths, inplace_safe/chunking memory optimizations, and
enable_efficient_fusion are not implemented. The structural-token attention
bias (from a separate StructuralTokenExpander) is accepted as an optional
`extra_attn_bias` argument on `DiffusionModule.__call__`, default None.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jopendde.backend import AbstractFromTorch, Linear, LayerNorm, register_from_torch
from jopendde.embedders import FourierEmbedding, RelativePositionEncoding
from jopendde.features import Features
from jopendde.primitives import Transition
from jopendde.transformer import AtomAttentionDecoder, AtomAttentionEncoder, DiffusionTransformer


# ---------------------------------------------------------------------------
# DiffusionConditioning (Algorithm 21 in AF3)
# ---------------------------------------------------------------------------


@register_from_torch("opendde.model.modules.diffusion.DiffusionConditioning")
class DiffusionConditioning(AbstractFromTorch):
    relpe: RelativePositionEncoding
    layernorm_z_trunk: LayerNorm
    linear_no_bias_z_trunk: Linear
    layernorm_z: LayerNorm
    linear_no_bias_z: Linear
    transition_z1: Transition
    transition_z2: Transition
    layernorm_s: LayerNorm
    linear_no_bias_s: Linear
    fourier_embedding: FourierEmbedding
    layernorm_n: LayerNorm
    linear_no_bias_n: Linear
    transition_s1: Transition
    transition_s2: Transition
    sigma_data: float

    # compress_pair_z is True in the opendde_v1 checkpoint (so layernorm_z_trunk/
    # linear_no_bias_z_trunk always exist and always run), hardcoded below;
    # from_torch asserts it.
    @classmethod
    def from_torch(cls, model):
        assert bool(model.compress_pair_z), (
            "jopendde hardcodes DiffusionConditioning compress_pair_z=True "
            "(the opendde_v1 setting); got compress_pair_z=False"
        )
        return super().from_torch(model)

    def prepare_cache(self, relp_feature: Float[Array, "N N F"], z_trunk: Float[Array, "N N Cz"]) -> Float[Array, "N N Czd"]:
        z_pair_trunk = self.linear_no_bias_z_trunk(self.layernorm_z_trunk(z_trunk))
        pair_z = jnp.concatenate([z_pair_trunk, self.relpe(relp_feature)], axis=-1)
        pair_z = self.linear_no_bias_z(self.layernorm_z(pair_z))
        pair_z = pair_z + self.transition_z1(pair_z)
        pair_z = pair_z + self.transition_z2(pair_z)
        return pair_z

    def __call__(
        self,
        t_hat_noise_level: Float[Array, "... N_sample"],
        relp_feature: Float[Array, "N N F"],
        s_inputs: Float[Array, "N Cs_inputs"],
        s_trunk: Float[Array, "N Cs"],
        z_trunk: Float[Array, "N N Cz"],
        pair_z: Float[Array, "N N Czd"] | None = None,
        use_conditioning: bool = True,
        **_ignored,  # perf-only kwargs; unused
    ):
        if pair_z is None:
            if not use_conditioning:
                s_trunk = 0 * s_trunk
                z_trunk = 0 * z_trunk
            pair_z = self.prepare_cache(relp_feature, z_trunk)

        single_s = jnp.concatenate([s_trunk, s_inputs], axis=-1)
        single_s = self.linear_no_bias_s(self.layernorm_s(single_s))
        noise_ratio = jnp.clip(t_hat_noise_level / self.sigma_data, 1e-10, None)
        noise_n = self.fourier_embedding(jnp.log(noise_ratio) / 4).astype(single_s.dtype)
        single_s = single_s[..., None, :, :] + self.linear_no_bias_n(self.layernorm_n(noise_n))[..., None, :]
        single_s = single_s + self.transition_s1(single_s)
        single_s = single_s + self.transition_s2(single_s)
        return single_s, pair_z


# ---------------------------------------------------------------------------
# DiffusionModule (Algorithm 20 in AF3)
# ---------------------------------------------------------------------------


@register_from_torch("opendde.model.modules.diffusion.DiffusionModule")
class DiffusionModule(AbstractFromTorch):
    diffusion_conditioning: DiffusionConditioning
    atom_attention_encoder: AtomAttentionEncoder
    layernorm_s: LayerNorm
    linear_no_bias_s: Linear
    diffusion_transformer: DiffusionTransformer
    layernorm_a: LayerNorm
    atom_attention_decoder: AtomAttentionDecoder
    normalize: LayerNorm
    sigma_data: float

    def f_forward(
        self,
        r_noisy: Float[Array, "... N_sample N_atom 3"],
        t_hat_noise_level: Float[Array, "... N_sample"],
        feat: Features,
        s_inputs: Float[Array, "N_token Cs_inputs"],
        s_trunk: Float[Array, "N_token Cs"],
        z_trunk: Float[Array, "N_token N_token Cz"],
        pair_z: Float[Array, "N_token N_token Czd"] | None = None,
        p_lm: Float[Array, "..."] | None = None,
        c_l: Float[Array, "..."] | None = None,
        use_conditioning: bool = True,
        extra_attn_bias: Float[Array, "..."] | None = None,
        token_pair_bias: Float[Array, "..."] | None = None,
    ) -> Float[Array, "... N_sample N_atom 3"]:
        s_single, z_pair = self.diffusion_conditioning(
            t_hat_noise_level,
            feat.relp,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            use_conditioning=use_conditioning,
        )

        s_trunk_expanded = s_trunk[..., None, :, :]  # insert N_sample=1 axis

        a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(
            feat.atom_to_token_idx,
            feat.ref_pos,
            feat.ref_charge,
            feat.ref_mask,
            feat.ref_atom_name_chars,
            feat.ref_element,
            feat.d_lm,
            feat.v_lm,
            feat.pad_info,
            r_l=r_noisy,
            s=s_trunk_expanded,
            z=z_pair,
            p_lm=p_lm,
            c_l=c_l,
        )

        a_token = a_token + self.linear_no_bias_s(self.layernorm_s(s_single))

        a_token = self.diffusion_transformer(
            a=a_token,
            s=s_single,
            z=z_pair,
            extra_attn_bias=extra_attn_bias,
            pair_bias_stack=token_pair_bias,
        )
        a_token = self.layernorm_a(a_token)

        r_update = self.atom_attention_decoder(
            atom_to_token_idx=feat.atom_to_token_idx,
            a=a_token,
            q_skip=q_skip,
            c_skip=c_skip,
            p_skip=p_skip,
        )
        return r_update

    def __call__(
        self,
        x_noisy: Float[Array, "... N_sample N_atom 3"],
        t_hat_noise_level: Float[Array, "... N_sample"],
        feat: Features,
        s_inputs: Float[Array, "N_token Cs_inputs"],
        s_trunk: Float[Array, "N_token Cs"],
        z_trunk: Float[Array, "N_token N_token Cz"],
        pair_z: Float[Array, "N_token N_token Czd"] | None = None,
        p_lm: Float[Array, "..."] | None = None,
        c_l: Float[Array, "..."] | None = None,
        use_conditioning: bool = True,
        extra_attn_bias: Float[Array, "..."] | None = None,
        token_pair_bias: Float[Array, "..."] | None = None,
        **_ignored,  # perf/FoldCP-only kwargs; unused
    ) -> Float[Array, "... N_sample N_atom 3"]:
        r_noisy = x_noisy / jnp.sqrt(self.sigma_data**2 + t_hat_noise_level**2)[..., None, None]

        r_update = self.f_forward(
            r_noisy=r_noisy,
            t_hat_noise_level=t_hat_noise_level,
            feat=feat,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            use_conditioning=use_conditioning,
            extra_attn_bias=extra_attn_bias,
            token_pair_bias=token_pair_bias,
        )

        s_ratio = (t_hat_noise_level / self.sigma_data)[..., None, None].astype(r_update.dtype)
        x_denoised = (
            1 / (1 + s_ratio**2) * x_noisy
            + t_hat_noise_level[..., None, None] / jnp.sqrt(1 + s_ratio**2) * r_update
        ).astype(r_update.dtype)
        return x_denoised


# ---------------------------------------------------------------------------
# centre_random_augmentation (Algorithm 19 in AF3).
#
# scipy's Rotation.random isn't jittable; use the standard
# normalized-4D-Gaussian-quaternion construction instead (same Haar-uniform
# SO(3) distribution, jittable).
# ---------------------------------------------------------------------------


def _random_quaternions(shape: tuple[int, ...], key) -> Float[Array, "... 4"]:
    o = jax.random.normal(key, shape + (4,))
    s = jnp.sum(o * o, axis=-1)
    sign = jnp.where(o[..., 0] < 0, -1.0, 1.0)
    return o / (sign * jnp.sqrt(s))[..., None]


def quaternion_to_matrix(quaternions: Float[Array, "... 4"]) -> Float[Array, "... 3 3"]:
    r, i, j, k = jnp.moveaxis(quaternions, -1, 0)
    two_s = 2.0 / jnp.sum(quaternions * quaternions, axis=-1)
    o = jnp.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def random_rotations(shape: tuple[int, ...], key) -> Float[Array, "... 3 3"]:
    return quaternion_to_matrix(_random_quaternions(shape, key))


def centre_random_augmentation(
    x_input_coords: Float[Array, "... N_atom 3"],
    key,
    N_sample: int = 1,
    s_trans: float = 1.0,
    centre_only: bool = False,
    mask: Float[Array, "... N_atom"] | None = None,
    eps: float = 1e-12,
) -> Float[Array, "... N_sample N_atom 3"]:
    if mask is None:
        x_input_coords = x_input_coords - jnp.mean(x_input_coords, axis=-2, keepdims=True)
    else:
        denom = jnp.sum(mask, axis=-1, keepdims=True) + eps
        center = jnp.sum(x_input_coords * mask[..., None], axis=-2) / denom
        x_input_coords = x_input_coords - center[..., None, :]

    x_input_coords = jnp.broadcast_to(
        x_input_coords[..., None, :, :],
        x_input_coords.shape[:-2] + (N_sample,) + x_input_coords.shape[-2:],
    )
    if centre_only:
        return x_input_coords

    batch_shape = x_input_coords.shape[:-3]
    key_r, key_t = jax.random.split(key)
    rot = random_rotations(batch_shape + (N_sample,), key_r)  # [..., N_sample, 3, 3]
    trans = s_trans * jax.random.normal(key_t, batch_shape + (N_sample, 3))

    x_augment = jnp.einsum("...ij,...mj->...mi", rot, x_input_coords) + trans[..., None, :]

    if mask is not None:
        x_augment = x_augment * mask[..., None, :, None]
    return x_augment


# ---------------------------------------------------------------------------
# Diffusion sampling loop (Algorithm 18 in AF3).
#
# The loop uses only a plain Euler predictor step -- no weighted-rigid-align /
# alignment-reverse-diff step.
# ---------------------------------------------------------------------------


def noise_schedule(
    num_steps: int, s_max: float = 160.0, s_min: float = 4e-4, rho: float = 7.0, sigma_data: float = 16.0
) -> Float[Array, "num_steps_plus_1"]:
    step_size = 1.0 / num_steps
    idx = jnp.arange(num_steps + 1, dtype=jnp.float32)
    t = (
        sigma_data
        * (s_max ** (1 / rho) + idx * step_size * (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho
    )
    return t.at[-1].set(0.0)


def sample_diffusion(
    diffusion_module: DiffusionModule,
    feat: Features,
    s_inputs: Float[Array, "N_token Cs_inputs"],
    s_trunk: Float[Array, "N_token Cs"],
    z_trunk: Float[Array, "N_token N_token Cz"],
    schedule: Float[Array, "num_steps_plus_1"],
    key,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    extra_attn_bias=None,
) -> Float[Array, "... N_sample N_atom 3"]:
    """Diffusion sampling loop. Single-chunk path (no diffusion_chunk_size
    splitting, no training-free-guidance path)."""

    N_atom = feat.atom_to_token_idx.shape[-1]
    batch_shape = s_inputs.shape[:-2]

    key, init_key = jax.random.split(key)
    x_l0 = schedule[0] * jax.random.normal(init_key, batch_shape + (N_sample, N_atom, 3))

    sigma_tm = schedule[:-1]
    sigma_t = schedule[1:]
    gammas = jnp.where(sigma_t > gamma_min, gamma0, 0.0)

    # `pair_z` and the atom-encoder's `p_lm`/base `c_l` depend only on
    # relp/z_trunk/ref-features, not on the noise level or noisy coords, so
    # they're loop-invariant. XLA does not hoist invariants out of a scan body,
    # so compute them once here and pass them into every step. (r_l=True is the
    # non-None gate for the token-pair-context term; the r_l value is unused by
    # prepare_cache.)
    pair_z = diffusion_module.diffusion_conditioning.prepare_cache(feat.relp, z_trunk)
    # One level deeper: each token DiffusionTransformer block reprojects its
    # attention bias from the loop-invariant `pair_z` (+ extra_attn_bias) every
    # step. Hoist that projection out and thread the precomputed stack into each
    # step.
    token_pair_bias = diffusion_module.diffusion_transformer.precompute_pair_bias(
        pair_z, extra_attn_bias
    )
    p_lm, c_l = diffusion_module.atom_attention_encoder.prepare_cache(
        ref_pos=feat.ref_pos,
        ref_charge=feat.ref_charge,
        ref_mask=feat.ref_mask,
        ref_atom_name_chars=feat.ref_atom_name_chars,
        ref_element=feat.ref_element,
        atom_to_token_idx=feat.atom_to_token_idx,
        d_lm=feat.d_lm,
        v_lm=feat.v_lm,
        pad_info=feat.pad_info,
        r_l=True,
        z=pair_z,
    )

    @jax.checkpoint
    def body_fn(carry, xs):
        x_l, step_key = carry
        c_tau_last, c_tau, gamma = xs
        step_key, aug_key, noise_key = jax.random.split(step_key, 3)

        x_l = centre_random_augmentation(x_l, key=aug_key, N_sample=1)[..., 0, :, :]

        t_hat = c_tau_last * (gamma + 1)
        delta_noise_level = jnp.sqrt(jnp.clip(t_hat**2 - c_tau_last**2, 0.0, None))
        x_noisy = x_l + noise_scale_lambda * delta_noise_level * jax.random.normal(noise_key, x_l.shape)

        t_hat_b = jnp.broadcast_to(t_hat, batch_shape + (N_sample,)).astype(x_l.dtype)

        x_denoised = diffusion_module(
            x_noisy=x_noisy,
            t_hat_noise_level=t_hat_b,
            feat=feat,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            extra_attn_bias=extra_attn_bias,
            token_pair_bias=token_pair_bias,
        )

        delta = (x_noisy - x_denoised) / t_hat_b[..., None, None]
        dt = c_tau - t_hat
        x_l_next = x_noisy + step_scale_eta * dt[..., None, None] * delta
        return (x_l_next, step_key), None

    (x_l, _), _ = jax.lax.scan(body_fn, (x_l0, key), (sigma_tm, sigma_t, gammas))
    return x_l
