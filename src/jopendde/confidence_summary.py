"""Confidence summary post-processing (pTM / ipTM / pLDDT / gPDE / chain-pair
metrics / AF3 clash / ranking score).

Turns the confidence head's logits + predicted coordinates into the
human-facing summary dict. Runs eagerly (not under jit): the chain-based
metrics loop over a data-dependent number of chains and use boolean-mask
indexing, both of which need concrete shapes. It's cheap scalar work on
already-computed model outputs, run once per prediction.

Bin parameters and the clash threshold are passed in explicitly rather than
importing the config object. The optional `interested_atom_mask` / vdw-clash /
`return_full_data` branches are not implemented.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

BinParams = tuple[float, float, int]  # (min_bin, max_bin, no_bins)


def get_bin_centers(min_bin: float, max_bin: float, no_bins: int) -> jnp.ndarray:
    bin_width = (max_bin - min_bin) / no_bins
    boundaries = jnp.linspace(min_bin, max_bin - bin_width, no_bins)
    return boundaries + 0.5 * bin_width


def logits_to_score(logits, bins: BinParams, return_prob: bool = False):
    prob = jnp.asarray(_softmax(logits))
    centers = get_bin_centers(*bins)
    score = prob @ centers
    return (score, prob) if return_prob else score


def _softmax(x):
    x = jnp.asarray(x)
    x = x - jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(x)
    return e / jnp.sum(e, axis=-1, keepdims=True)


def calculate_normalization(N: int) -> float:
    return 1.24 * (max(N, 19) - 15) ** (1 / 3) - 1.8


def _ptm_bin_weight(bins: BinParams, ptm_norm: float) -> jnp.ndarray:
    centers = get_bin_centers(*bins)
    return 1.0 / (1.0 + (centers / ptm_norm) ** 2)


def calculate_ptm(pae_prob, has_frame, bins: BinParams, token_mask=None) -> jnp.ndarray:
    has_frame = np.asarray(has_frame).astype(bool)
    pae_prob = jnp.asarray(pae_prob)
    if token_mask is not None:
        token_mask = np.asarray(token_mask).astype(bool)
        pae_prob = pae_prob[..., token_mask, :, :][..., :, token_mask, :]
        has_frame = has_frame[token_mask]
    if has_frame.sum() == 0:
        return jnp.zeros(pae_prob.shape[:-3])
    N_d = has_frame.shape[-1]
    w = _ptm_bin_weight(bins, calculate_normalization(N_d))
    token_token_ptm = jnp.sum(pae_prob * w, axis=-1)  # [..., N_d, N_d]
    per_token = jnp.mean(token_token_ptm, axis=-1)  # [..., N_d]
    return jnp.max(per_token[..., has_frame], axis=-1)


def calculate_iptm(pae_prob, has_frame, asym_id, bins: BinParams, token_mask=None, eps=1e-8):
    has_frame = np.asarray(has_frame).astype(bool)
    asym_id = np.asarray(asym_id).astype(np.int64)
    pae_prob = jnp.asarray(pae_prob)
    if token_mask is not None:
        token_mask = np.asarray(token_mask).astype(bool)
        pae_prob = pae_prob[..., token_mask, :, :][..., :, token_mask, :]
        has_frame = has_frame[token_mask]
        asym_id = asym_id[token_mask]
    if has_frame.sum() == 0:
        return jnp.zeros(pae_prob.shape[:-3])
    N_d = has_frame.shape[-1]
    w = _ptm_bin_weight(bins, calculate_normalization(N_d))
    token_token_ptm = jnp.sum(pae_prob * w, axis=-1)  # [..., N_d, N_d]
    is_diff_chain = jnp.asarray(asym_id[None, :] != asym_id[:, None])  # [N_d, N_d]
    iptm = jnp.sum(token_token_ptm * is_diff_chain, axis=-1) / (
        eps + jnp.sum(is_diff_chain, axis=-1)
    )
    return jnp.max(iptm[..., has_frame], axis=-1)


def _remap_asym(asym_id: np.ndarray) -> np.ndarray:
    uniq = np.unique(asym_id)
    if len(uniq) != asym_id.max() + 1:
        remap = {int(o): n for n, o in enumerate(uniq)}
        asym_id = np.array([remap[int(x)] for x in asym_id], dtype=np.int64)
    return asym_id


def calculate_chain_based_ptm(pae_prob, has_frame, asym_id, token_is_ligand, bins) -> dict:
    has_frame = np.asarray(has_frame).astype(bool)
    asym_id = _remap_asym(np.asarray(asym_id).astype(np.int64))
    token_is_ligand = np.asarray(token_is_ligand).astype(bool)
    masks = {aid: asym_id == aid for aid in np.unique(asym_id)}
    N_chain = len(masks)
    chain_is_ligand = {
        aid: token_is_ligand[m].sum() >= m.sum() // 2 for aid, m in masks.items()
    }
    batch = jnp.asarray(pae_prob).shape[:-3]

    chain_pair_iptm = np.zeros(batch + (N_chain, N_chain), dtype=np.float32)
    for a in range(N_chain):
        for b in range(N_chain):
            if a == b:
                continue
            if a > b:
                chain_pair_iptm[..., a, b] = chain_pair_iptm[..., b, a]
                continue
            chain_pair_iptm[..., a, b] = np.asarray(
                calculate_iptm(pae_prob, has_frame, asym_id, bins, token_mask=masks[a] + masks[b])
            )

    chain_ptm = np.zeros(batch + (N_chain,), dtype=np.float32)
    for aid, m in masks.items():
        chain_ptm[..., aid] = np.asarray(calculate_ptm(pae_prob, has_frame, bins, token_mask=m))

    chain_has_frame = [bool((masks[i] & has_frame).any()) for i in range(N_chain)]
    chain_iptm = np.zeros(batch + (N_chain,), dtype=np.float32)
    for aid in range(N_chain):
        pairs = [
            (i, j)
            for i in range(N_chain)
            for j in range(N_chain)
            if (i == aid or j == aid) and i != j and chain_has_frame[i]
        ]
        if pairs:
            chain_iptm[..., aid] = np.stack(
                [chain_pair_iptm[..., i, j] for (i, j) in pairs], axis=-1
            ).mean(axis=-1)

    chain_pair_iptm_global = np.zeros(batch + (N_chain, N_chain), dtype=np.float32)
    for a in range(N_chain):
        for b in range(N_chain):
            if a == b:
                continue
            if chain_is_ligand[a]:
                chain_pair_iptm_global[..., a, b] = chain_iptm[..., a]
            elif chain_is_ligand[b]:
                chain_pair_iptm_global[..., a, b] = chain_iptm[..., b]
            else:
                chain_pair_iptm_global[..., a, b] = (chain_iptm[..., a] + chain_iptm[..., b]) * 0.5

    return {
        "chain_ptm": jnp.asarray(chain_ptm),
        "chain_iptm": jnp.asarray(chain_iptm),
        "chain_pair_iptm": jnp.asarray(chain_pair_iptm),
        "chain_pair_iptm_global": jnp.asarray(chain_pair_iptm_global),
    }


def calculate_chain_based_gpde(token_pair_pde, contact_probs, asym_id, eps=1e-8) -> dict:
    asym_id = _remap_asym(np.asarray(asym_id).astype(np.int64))
    N_chain = len(np.unique(asym_id))
    pde = jnp.asarray(token_pair_pde)
    cp = jnp.asarray(contact_probs)
    batch = pde.shape[:-2]

    def _gpde(m1, m2):
        mc = cp[..., m1, :][..., m2]
        mp = pde[..., m1, :][..., m2]
        return jnp.sum(mp * mc, axis=(-1, -2)) / (jnp.sum(mc, axis=(-1, -2)) + eps)

    chain_gpde = np.zeros(batch + (N_chain,), dtype=np.float32)
    for a in range(N_chain):
        chain_gpde[..., a] = np.asarray(_gpde(asym_id == a, asym_id == a))
    chain_pair_gpde = np.zeros(batch + (N_chain, N_chain), dtype=np.float32)
    for a in range(N_chain):
        for b in range(N_chain):
            if a == b:
                continue
            if b < a:
                chain_pair_gpde[..., a, b] = chain_pair_gpde[..., b, a]
                continue
            chain_pair_gpde[..., a, b] = np.asarray(_gpde(asym_id == a, asym_id == b))
    return {"chain_gpde": jnp.asarray(chain_gpde), "chain_pair_gpde": jnp.asarray(chain_pair_gpde)}


def calculate_chain_based_plddt(atom_plddt, asym_id, atom_to_token_idx) -> dict:
    asym_id = _remap_asym(np.asarray(asym_id).astype(np.int64))
    a2t = np.asarray(atom_to_token_idx).astype(np.int64)
    ap = jnp.asarray(atom_plddt)
    masks = {aid: asym_id == aid for aid in np.unique(asym_id)}
    N_chain = len(masks)
    batch = ap.shape[:-1]

    def _lddt(tok_mask):
        return jnp.mean(ap[:, tok_mask[a2t]], axis=-1)

    chain_plddt = np.zeros(batch + (N_chain,), dtype=np.float32)
    for aid, m in masks.items():
        chain_plddt[..., aid] = np.asarray(_lddt(m))
    chain_pair_plddt = np.zeros(batch + (N_chain, N_chain), dtype=np.float32)
    for a in range(N_chain):
        for b in range(N_chain):
            if a == b:
                continue
            chain_pair_plddt[..., a, b] = np.asarray(_lddt(masks[a] + masks[b]))
    return {"chain_plddt": jnp.asarray(chain_plddt), "chain_pair_plddt": jnp.asarray(chain_pair_plddt)}


def calculate_af3_has_clash(coord, asym_id, atom_to_token_idx, atom_is_polymer, threshold=1.1):
    """AF3 steric clash: for each polymer-polymer chain pair, count inter-chain
    atom pairs closer than `threshold` A; a pair clashes if >100 such contacts
    or >0.5 relative to the smaller chain. Returns [N_sample] bool (any pair)."""
    coord = np.asarray(coord)
    asym_id = _remap_asym(np.asarray(asym_id).astype(np.int64))
    a2t = np.asarray(atom_to_token_idx).astype(np.int64)
    is_poly = np.asarray(atom_is_polymer).astype(bool)
    atom_asym = asym_id[a2t]  # [N_atom]
    N_sample = coord.shape[0]
    chains = np.unique(asym_id)
    # a chain is polymer if any of its atoms is polymer (AF3 clash: polymers only)
    chain_poly = {int(c): bool(is_poly[atom_asym == c].any()) for c in chains}
    out = np.zeros(N_sample, dtype=bool)
    for s in range(N_sample):
        clashed = False
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                ci, cj = int(chains[i]), int(chains[j])
                if not (chain_poly[ci] and chain_poly[cj]):
                    continue
                mi, mj = atom_asym == ci, atom_asym == cj
                d = np.linalg.norm(coord[s, mi][:, None, :] - coord[s, mj][None, :, :], axis=-1)
                total = int((d < threshold).sum())
                rel = total / max(1, min(mi.sum(), mj.sum()))
                if total > 100 or rel > 0.5:
                    clashed = True
        out[s] = clashed
    return jnp.asarray(out.astype(np.float32))


def compute_summary_confidence(
    *,
    pae_logits,
    plddt_logits,
    pde_logits,
    contact_probs,
    token_asym_id,
    token_has_frame,
    atom_coordinate,
    atom_to_token_idx,
    atom_is_polymer,
    N_recycle: int,
    plddt_bins: BinParams,
    pae_bins: BinParams,
    pde_bins: BinParams,
    af3_clash_threshold: float = 1.1,
) -> list[dict[str, Any]]:
    """Compute the per-sample summary confidence dict (one dict per sample)."""
    a2t = np.asarray(atom_to_token_idx).astype(np.int64)
    asym = np.asarray(token_asym_id).astype(np.int64)
    is_poly = np.asarray(atom_is_polymer)
    contact_probs = jnp.asarray(contact_probs)

    atom_is_ligand = (1 - is_poly).astype(np.int64)
    token_is_ligand = np.zeros_like(asym)
    np.add.at(token_is_ligand, a2t, atom_is_ligand)
    token_is_ligand = token_is_ligand > 0

    atom_plddt = logits_to_score(plddt_logits, plddt_bins)  # [N_s, N_atom]
    token_pair_pde = logits_to_score(pde_logits, pde_bins)  # [N_s, N_tok, N_tok]
    _, pae_prob = logits_to_score(pae_logits, pae_bins, return_prob=True)

    if contact_probs.ndim == 2:
        contact_probs_b = jnp.broadcast_to(contact_probs, atom_plddt.shape[:1] + contact_probs.shape)
    else:
        contact_probs_b = contact_probs

    summary = {}
    summary["plddt"] = jnp.mean(atom_plddt, axis=-1) * 100
    summary["gpde"] = jnp.sum(token_pair_pde * contact_probs_b, axis=(-1, -2)) / jnp.sum(
        contact_probs_b, axis=(-1, -2)
    )
    summary["ptm"] = calculate_ptm(pae_prob, token_has_frame, pae_bins)
    summary["iptm"] = calculate_iptm(pae_prob, token_has_frame, asym, pae_bins)
    summary.update(calculate_chain_based_gpde(token_pair_pde, contact_probs_b, asym))
    summary.update(
        calculate_chain_based_ptm(pae_prob, token_has_frame, asym, token_is_ligand, pae_bins)
    )
    summary.update(calculate_chain_based_plddt(atom_plddt, asym, a2t))
    summary["has_clash"] = calculate_af3_has_clash(
        atom_coordinate, asym, a2t, is_poly, af3_clash_threshold
    )
    summary["disorder"] = jnp.zeros_like(summary["ptm"])
    summary["ranking_score"] = (
        0.8 * summary["iptm"]
        + 0.2 * summary["ptm"]
        + 0.5 * summary["disorder"]
        - 100 * summary["has_clash"]
    )

    def _native(x):
        x = np.asarray(x)
        return x.item() if x.ndim == 0 else x.tolist()

    N_sample = int(atom_plddt.shape[0])
    per_sample = []
    for i in range(N_sample):
        d = {}
        for k, v in summary.items():
            v = np.asarray(v)
            vi = v[i] if v.shape[:1] == (N_sample,) else v
            d[k] = _native(vi)  # JSON-native (float / nested list) for dumping
        d["num_recycles"] = int(N_recycle)
        per_sample.append(d)
    return per_sample
