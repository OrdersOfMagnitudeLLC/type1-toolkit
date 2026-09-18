# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

"""
NSMace session 4: numpy prototype of the full MACE forward pass for water,
including the correct conv_tp tensor product, symmetric-contraction products
block, skip_tp residual, atomic energies, and scale/shift.

Verified against live MACE model hook outputs and final energy before porting
to C++. Run: python3 reference/numpy_prototype.py
"""
import json
import numpy as np

WEIGHTS_JSON = "../weights/mace_mp_small_weights.json"
CG_NPZ = "../weights/cg_u_matrices.npz"

R_MAX = 6.0
NUM_BESSEL = 10
AVG_NUM_NEIGHBORS = 61.964672446250916
SCALE = 0.8041538754478097
SHIFT = 0.16409696359187365
ATOMIC_ENERGY_O = -7.28459863421322
ATOMIC_ENERGY_H = -3.667168021358939
SILU_NORM2MOM = 1.6791767923989418

with open(WEIGHTS_JSON) as f:
    raw = json.load(f)

W = {}
for k, v in raw.items():
    arr = np.array(v["data"] if isinstance(v, dict) and "data" in v else v, dtype=np.float64)
    W[k] = arr

cg = np.load(CG_NPZ)
U1, U2, U3 = cg["U1"], cg["U2"], cg["U3"]  # (16,1) (16,16,4) (16,16,16,23)


def silu(x):
    return x / (1.0 + np.exp(-x)) * SILU_NORM2MOM


def poly_envelope(r, r_max, p=5):
    if r >= r_max:
        return 0.0
    x = r / r_max
    return (1.0 - (p + 1) * (p + 2) / 2.0 * x**p
            + p * (p + 2) * x**(p + 1)
            - p * (p + 1) / 2.0 * x**(p + 2))


def bessel_basis(r, r_max=R_MAX, num_bessel=NUM_BESSEL):
    out = np.zeros(num_bessel)
    if r < 1e-8 or r >= r_max:
        return out
    env = poly_envelope(r, r_max)
    norm = np.sqrt(2.0 / r_max)
    for n in range(1, num_bessel + 1):
        out[n - 1] = norm * np.sin(n * np.pi * r / r_max) / r * env
    return out


def spherical_harmonics(dx, dy, dz, r, max_ell=3):
    ir = 1.0 / r if r > 1e-10 else 0.0
    x, y, z = dx * ir, dy * ir, dz * ir
    Y = [1.0]
    if max_ell < 1:
        return np.array(Y)
    sh_1_0, sh_1_1, sh_1_2 = x, y, z
    n1 = np.sqrt(3.0)
    Y += [sh_1_0 * n1, sh_1_1 * n1, sh_1_2 * n1]
    if max_ell < 2:
        return np.array(Y)
    sh_2_0 = np.sqrt(3.0) * x * z
    sh_2_1 = np.sqrt(3.0) * x * y
    y2 = y * y
    x2z2 = x * x + z * z
    sh_2_2 = y2 - 0.5 * x2z2
    sh_2_3 = np.sqrt(3.0) * y * z
    sh_2_4 = np.sqrt(3.0) / 2.0 * (z * z - x * x)
    n2 = np.sqrt(5.0)
    Y += [sh_2_0 * n2, sh_2_1 * n2, sh_2_2 * n2, sh_2_3 * n2, sh_2_4 * n2]
    if max_ell < 3:
        return np.array(Y)
    sh_3_0 = np.sqrt(5.0 / 6.0) * (sh_2_0 * z + sh_2_4 * x)
    sh_3_1 = np.sqrt(5.0) * sh_2_0 * y
    sh_3_2 = np.sqrt(3.0 / 8.0) * (4.0 * y2 - x2z2) * x
    sh_3_3 = 0.5 * y * (2.0 * y2 - 3.0 * x2z2)
    sh_3_4 = np.sqrt(3.0 / 8.0) * z * (4.0 * y2 - x2z2)
    sh_3_5 = np.sqrt(5.0) * sh_2_4 * y
    sh_3_6 = np.sqrt(5.0 / 6.0) * (sh_2_4 * z - sh_2_0 * x)
    n3 = np.sqrt(7.0)
    Y += [sh_3_0 * n3, sh_3_1 * n3, sh_3_2 * n3, sh_3_3 * n3, sh_3_4 * n3, sh_3_5 * n3, sh_3_6 * n3]
    return np.array(Y)


def radial_mlp_forward(x, interaction_idx):
    arch = [10, 64, 64, 64, 512]
    buf = x
    for k in range(4):
        Wk = W[f"interactions.{interaction_idx}.conv_tp_weights.layer{k}.weight"].reshape(arch[k], arch[k + 1])
        buf = (buf @ Wk) / np.sqrt(arch[k])
        if k < 3:
            buf = silu(buf)
    return buf  # [512]


def e3nn_linear(x, weight_flat, fan_in, fan_out):
    Wm = weight_flat.reshape(fan_in, fan_out)
    return (x @ Wm) / np.sqrt(fan_in)


def node_embed(atom_type):
    Wm = W["node_embedding.linear.weight"].reshape(89, 128)
    return Wm[atom_type, :] / np.sqrt(89.0)


L_DIMS = [1, 3, 5, 7]
L_OFFSETS = [0, 128, 512, 1152]  # offsets into 2048-d irreps_mid


def conv_tp(node_feats_up_j, tp_weights, sh):
    """node_feats_up_j: [128], tp_weights: [512], sh: [16] -> mji [2048]"""
    mji = np.zeros(2048)
    for l in range(4):
        dim = L_DIMS[l]
        off = L_OFFSETS[l]
        w_block = tp_weights[l * 128:(l + 1) * 128]          # [128]
        sh_block = sh[sum(L_DIMS[:l]):sum(L_DIMS[:l]) + dim]  # [dim]
        block = np.outer(w_block * node_feats_up_j, sh_block)  # [128, dim]
        mji[off:off + 128 * dim] = block.reshape(-1)
    return mji


def interaction_linear(message, interaction_idx):
    Wflat = W[f"interactions.{interaction_idx}.linear.weight"]
    out = np.zeros(2048)
    for l in range(4):
        dim = L_DIMS[l]
        off = L_OFFSETS[l]
        Wl = Wflat[l * 16384:(l + 1) * 16384].reshape(128, 128)
        block = message[off:off + 128 * dim].reshape(128, dim)
        out_block = (Wl.T @ block) / np.sqrt(128.0)
        out[off:off + 128 * dim] = out_block.reshape(-1)
    return out


def skip_tp(node_feats, elem_idx, interaction_idx):
    Wflat = W[f"interactions.{interaction_idx}.skip_tp.weight"].reshape(128, 89, 128)
    pw = 1.0 / np.sqrt(128 * 89)
    return pw * (node_feats @ Wflat[:, elem_idx, :])


def symmetric_contraction(feat16x128, elem_idx, product_idx):
    """feat16x128: [128,16] channel-major -> out [128]"""
    feat = feat16x128.T  # [16,128] -> we want [c,i] i.e. feat16x128 already [c][i]? define below
    # We keep feat as [c=128, i=16]
    c_dim = 128
    Wmax = W[f"products.{product_idx}.symmetric_contractions.contractions.0.weights_max"].reshape(89, 23, 128)
    Wnu2 = W[f"products.{product_idx}.symmetric_contractions.contractions.0.weights.0"].reshape(89, 4, 128)
    Wnu1 = W[f"products.{product_idx}.symmetric_contractions.contractions.0.weights.1"].reshape(89, 1, 128)

    Wmax_e = Wmax[elem_idx]  # [23,128]
    Wnu2_e = Wnu2[elem_idx]  # [4,128]
    Wnu1_e = Wnu1[elem_idx]  # [1,128]

    feat_ci = feat16x128  # [c,i], i in 0..15

    # out3[c,w,x] = sum_{i,k} U3[w,x,i,k] * Wmax_e[k,c] * feat[c,i]
    # -> for each c: M[c] = einsum('wxik,k->wxi', U3, Wmax_e[:,c]) then dot with feat[c,:] over i
    out3 = np.einsum('wxik,kc,ci->cwx', U3, Wmax_e, feat_ci)  # [c,16,16]

    c_tensor2 = np.einsum('wxk,kc->cwx', U2, Wnu2_e) + out3  # [c,16,16]
    out2 = np.einsum('cwi,ci->cw', c_tensor2, feat_ci)  # [c,16]

    c_tensor1 = np.einsum('wk,kc->cw', U1, Wnu1_e) + out2  # [c,16]
    out1 = np.einsum('ci,ci->c', c_tensor1, feat_ci)  # [c]

    return out1


def readout_linear(h, idx):
    Wm = W[f"readouts.{idx}.linear.weight"]
    pw = 1.0 / np.sqrt(len(Wm))
    return float(np.dot(Wm, h) * pw)


def readout_nonlinear(h, idx):
    W1 = W[f"readouts.{idx}.linear_1.weight"].reshape(128, 16)
    W2 = W[f"readouts.{idx}.linear_2.weight"]
    mid = silu((h @ W1) / np.sqrt(128.0))
    return float(np.dot(mid, W2) / np.sqrt(16.0))


def main():
    positions = np.array([
        [0.0, 0.0, 0.119],
        [0.0, 0.763, -0.477],
        [0.0, -0.763, -0.477],
    ])
    elems = [7, 0, 0]  # O(idx7), H(idx0), H(idx0) -- model index = Z-1
    N = 3

    node_attrs_onehot = elems  # index form

    # neighbor list (full, cutoff 6.0, all pairs within cutoff, no self)
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = positions[j] - positions[i]
            r = np.linalg.norm(d)
            if r < R_MAX:
                neighbors[i].append((j, d, r))

    h = np.array([node_embed(e) for e in elems])  # [3,128]

    node_feats_list = []
    h_cur = h.copy()
    for inter_idx in range(2):
        # sc uses node_feats BEFORE linear_up, i.e. h_cur (pre-update)
        sc = np.array([skip_tp(h_cur[i], elems[i], inter_idx) for i in range(N)])

        node_feats_up = np.array([e3nn_linear(h_cur[i], W[f"interactions.{inter_idx}.linear_up.weight"], 128, 128) for i in range(N)])

        message = np.zeros((N, 2048))
        for i in range(N):
            for (j, d, r) in neighbors[i]:
                bessel = bessel_basis(r)
                tp_w = radial_mlp_forward(bessel, inter_idx)
                sh = spherical_harmonics(d[0], d[1], d[2], r)
                mji = conv_tp(node_feats_up[j], tp_w, sh)
                message[i] += mji

        message = np.array([interaction_linear(message[i], inter_idx) for i in range(N)])
        message /= AVG_NUM_NEIGHBORS

        # reshape to [N,128,16]
        feat16 = np.zeros((N, 128, 16))
        for i in range(N):
            for l in range(4):
                dim = L_DIMS[l]
                off = L_OFFSETS[l]
                ioff = sum(L_DIMS[:l])
                feat16[i, :, ioff:ioff + dim] = message[i, off:off + 128 * dim].reshape(128, dim)

        contracted = np.array([symmetric_contraction(feat16[i], elems[i], inter_idx) for i in range(N)])  # [N,128]
        products_lin = np.array([e3nn_linear(contracted[i], W[f"products.{inter_idx}.linear.weight"], 128, 128) for i in range(N)])
        node_feats_out = products_lin + sc  # [N,128]

        node_feats_list.append(node_feats_out)
        h_cur = node_feats_out

    # readouts
    e0_readout = np.array([readout_linear(node_feats_list[0][i], 0) for i in range(N)])
    e1_readout = np.array([readout_nonlinear(node_feats_list[1][i], 1) for i in range(N)])

    node_inter_es = e0_readout + e1_readout  # per-atom
    node_inter_es_scaled = SCALE * node_inter_es + SHIFT
    inter_e = np.sum(node_inter_es_scaled)

    atomic_e0 = sum(ATOMIC_ENERGY_O if e == 7 else ATOMIC_ENERGY_H for e in elems)

    total_energy = atomic_e0 + inter_e

    reference = -14.047703873269672
    print("products0_out[0][:3] =", node_feats_list[0][0][:3])
    print("products1_out[0][:3] =", node_feats_list[1][0][:3])
    print("atomic_e0:", atomic_e0)
    print("inter_e:", inter_e)
    print("E_total:", total_energy)
    print("Reference:", reference)
    print("Delta:", abs(total_energy - reference))


if __name__ == "__main__":
    main()
