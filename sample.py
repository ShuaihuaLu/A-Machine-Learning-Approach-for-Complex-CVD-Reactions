import os
import gc
import heapq
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from ase.io import iread, write
from ase.neighborlist import neighbor_list
from ase.data import chemical_symbols, covalent_radii, atomic_numbers
from e3nn import o3

EPS = 1e-12

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

try:
    torch.serialization.add_safe_globals([slice])
except AttributeError:
    pass


def _to_z_list(species):
    if species is None:
        return None
    out = []
    for s in species:
        if isinstance(s, (int, np.integer)):
            out.append(int(s))
        else:
            out.append(int(atomic_numbers[str(s)]))
    return out


def get_reactive_mask(atoms, z_surface_threshold=None, reactive_species=None, device="cpu"):
    """
    Define the chemically active region.
    Priority:
      1) tags if present and non-zero
      2) z_surface_threshold
      3) optional reactive_species filter
    """
    N = len(atoms)
    mask = np.ones(N, dtype=bool)

    tags = atoms.arrays.get("tags", None)
    if tags is not None and np.any(np.asarray(tags) != 0):
        mask = np.asarray(tags) > 0
    elif z_surface_threshold is not None:
        mask = np.asarray(atoms.positions[:, 2]) >= float(z_surface_threshold)

    z_list = _to_z_list(reactive_species)
    if z_list is not None:
        mask &= np.isin(np.asarray(atoms.numbers), np.asarray(z_list))

    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def quantile_signature(x, n_q=64):
    """
    Fixed-length 1D distribution signature.
    More robust than raw sorted vectors when the number of active atoms varies.
    """
    x = x.flatten().float()
    if x.numel() == 0:
        return torch.zeros(n_q, device=x.device)
    q = torch.linspace(0.0, 1.0, n_q, device=x.device)
    return torch.quantile(x, q)


def get_torch_graph_with_elements(
    atoms,
    rc=3.5,
    z_surface_threshold=None,
    reactive_species=None,
    bond_scale=1.25,
    bond_sharpness=0.10,
    element_types=None,
    device="cpu",
):
    """
    Build neighbor graph with a chemistry-aware soft bond gate.
    - cosine cutoff: finite support and smoothness
    - covalent-radius gate: favors chemically plausible bonds
    - optional surface suppression: reduces deep-slab thermal noise
    """
    i_list, j_list, d_list, D_list = neighbor_list("ijdD", atoms, rc)

    if element_types is None:
        element_types = np.unique(atoms.numbers)
    else:
        element_types = np.asarray(element_types, dtype=np.int64)

    num_elements = len(element_types)
    element_symbols = [chemical_symbols[int(z)] for z in element_types]

    if len(i_list) == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, 3, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            num_elements,
            element_types,
            element_symbols,
        )

    z_to_idx = {int(z): k for k, z in enumerate(element_types)}
    atom_z = np.asarray(atoms.numbers, dtype=np.int64)
    mapped_z_np = np.fromiter((z_to_idx[int(z)] for z in atom_z), dtype=np.int64, count=len(atom_z))

    edge_src = torch.as_tensor(i_list, dtype=torch.long, device=device)
    edge_dst = torch.as_tensor(j_list, dtype=torch.long, device=device)
    edge_vec = torch.as_tensor(D_list, dtype=torch.float32, device=device)
    edge_len = torch.as_tensor(d_list, dtype=torch.float32, device=device)
    mapped_z = torch.as_tensor(mapped_z_np, dtype=torch.long, device=device)
    edge_elem_idx = mapped_z[edge_dst]

    # Smooth cutoff
    r = torch.clamp(edge_len / rc, 0.0, 1.0)
    cutoff_w = 0.5 * (1.0 + torch.cos(torch.pi * r))

    # Soft chemistry gate based on covalent radii
    cov = torch.as_tensor([covalent_radii[int(z)] for z in atom_z], dtype=torch.float32, device=device)
    bond_cut = bond_scale * (cov[edge_src] + cov[edge_dst])
    bond_gate = torch.sigmoid((bond_cut - edge_len) / max(bond_sharpness, 1e-3))
    edge_weights = cutoff_w * (0.25 + 0.75 * bond_gate)

    # Optional substrate suppression: attenuate deep-slab edges
    if z_surface_threshold is not None:
        pos_z = torch.as_tensor(atoms.positions[:, 2], dtype=torch.float32, device=device)
        below = (pos_z[edge_src] < z_surface_threshold) & (pos_z[edge_dst] < z_surface_threshold)
        edge_weights = torch.where(below, edge_weights * 0.15, edge_weights)

    return (
        edge_src,
        edge_dst,
        edge_vec,
        edge_weights,
        edge_elem_idx,
        num_elements,
        element_types,
        element_symbols,
    )


@torch.inference_mode()
def compute_atomic_features(
    atoms,
    rc=3.5,
    z_surface_threshold=None,
    reactive_species=None,
    element_types=None,
    device=None,
    mp_steps=3,
    mp_weight=0.5,
):
    """
    Output:
      anomaly_scores: (N,)
      atomic_fp:      (N, F_atom)
      frame_fp:       (F_frame,)
      coord_total:    (N,)  soft coordination number
      reactive_mask:  (N,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = len(atoms)
    if N == 0:
        return (
            torch.tensor([], device=device),
            torch.tensor([], device=device),
            torch.tensor([], device=device),
            torch.tensor([], device=device),
            torch.tensor([], device=device, dtype=torch.bool),
        )

    (
        edge_src,
        edge_dst,
        edge_vec,
        edge_weights,
        edge_elem_idx,
        num_elements,
        element_types,
        element_symbols,
    ) = get_torch_graph_with_elements(
        atoms,
        rc=rc,
        z_surface_threshold=z_surface_threshold,
        reactive_species=reactive_species,
        element_types=element_types,
        device=device,
    )

    reactive_mask = get_reactive_mask(
        atoms,
        z_surface_threshold=z_surface_threshold,
        reactive_species=reactive_species,
        device=device,
    )

    if len(edge_src) == 0:
        max_entropy = np.log(1.0)
        anomaly_scores = torch.full((N,), max_entropy, device=device)
        atomic_fp = torch.zeros((N, num_elements * 4), device=device)
        frame_fp = torch.zeros((64,), device=device)
        coord_total = torch.zeros((N,), device=device)
        return anomaly_scores, atomic_fp, frame_fp, coord_total, reactive_mask

    # Degree for normalization
    deg = torch.zeros(N, dtype=torch.float32, device=device)
    deg.index_add_(0, edge_src, edge_weights)
    deg.index_add_(0, edge_dst, edge_weights)
    norm = edge_weights / torch.sqrt(deg[edge_src].clamp_min(EPS) * deg[edge_dst].clamp_min(EPS))

    # Soft coordination by species
    flat_node_idx = edge_src * num_elements + edge_elem_idx
    coord_species_flat = torch.zeros((N * num_elements,), dtype=torch.float32, device=device)
    coord_species_flat.index_add_(0, flat_node_idx, edge_weights)
    coord_species = coord_species_flat.view(N, num_elements)
    coord_total = coord_species.sum(dim=-1)

    # Base equivariant tensors
    sh_1o = o3.spherical_harmonics("1o", edge_vec, normalize=True)  # (E, 3)
    sh_2e = o3.spherical_harmonics("2e", edge_vec, normalize=True)  # (E, 5)

    w_1o = sh_1o * edge_weights.unsqueeze(-1)
    w_2e = sh_2e * edge_weights.unsqueeze(-1)

    # 1st-order tensor: outer product of vector harmonics
    edge_outer = torch.einsum("ei,ej->eij", w_1o, w_1o).reshape(-1, 9)
    edge_outer = 0.5 * (edge_outer + edge_outer.view(-1, 3, 3).transpose(-1, -2).reshape(-1, 9))

    # 2nd-order scalar proxy
    edge_2e_inv = (w_2e ** 2).sum(dim=-1, keepdim=True)

    node_flat_1o = torch.zeros((N * num_elements, 9), dtype=torch.float32, device=device)
    node_flat_1o.index_add_(0, flat_node_idx, edge_outer)
    base_tensors = node_flat_1o.view(N, num_elements, 3, 3)
    base_tensors = 0.5 * (base_tensors + base_tensors.transpose(-1, -2))

    node_flat_2e = torch.zeros((N * num_elements, 1), dtype=torch.float32, device=device)
    node_flat_2e.index_add_(0, flat_node_idx, edge_2e_inv)
    base_2e_inv = node_flat_2e.view(N, num_elements, 1)

    # Topological message passing
    H_1o = base_tensors.clone()
    H_2e = base_2e_inv.clone()

    node_norm = torch.zeros(N, dtype=torch.float32, device=device)
    node_norm.index_add_(0, edge_src, norm)

    for _ in range(mp_steps):
        msg_1o = H_1o[edge_dst] * norm.view(-1, 1, 1, 1)
        msg_2e = H_2e[edge_dst] * norm.view(-1, 1, 1)

        agg_1o = torch.zeros_like(H_1o)
        agg_1o.index_add_(0, edge_src, msg_1o)

        agg_2e = torch.zeros_like(H_2e)
        agg_2e.index_add_(0, edge_src, msg_2e)

        update_norm = node_norm.clamp_min(1.0).view(N, 1, 1, 1)
        update_norm_2 = node_norm.clamp_min(1.0).view(N, 1, 1)

        H_1o = (1.0 - mp_weight) * H_1o + mp_weight * (agg_1o / update_norm)
        H_2e = (1.0 - mp_weight) * H_2e + mp_weight * (agg_2e / update_norm_2)

    # Spectral fingerprint
    evals = torch.linalg.eigvalsh(H_1o)                  # (N, num_elements, 3)
    evals = torch.clamp_min(evals, EPS)
    evals_flat = evals.reshape(N, -1)                    # (N, num_elements*3)
    h2_flat = H_2e.reshape(N, -1)                        # (N, num_elements)

    atomic_fp = torch.cat([evals_flat, h2_flat], dim=-1) # (N, num_elements*4)

    # Anomaly score: spectrum entropy deficit
    p = evals_flat / evals_flat.sum(dim=-1, keepdim=True).clamp_min(EPS)
    entropy = -(p * torch.log(p)).sum(dim=-1)
    max_entropy = np.log(float(evals_flat.shape[-1]))
    anomaly_scores = max_entropy - entropy
    anomaly_scores = torch.where(torch.isfinite(anomaly_scores), anomaly_scores, torch.zeros_like(anomaly_scores))

    # Frame embedding: active-region pool + global pool + coordination statistics
    reactive_f = reactive_mask.float().unsqueeze(-1)
    reactive_count = reactive_f.sum().clamp_min(1.0)

    reactive_pool = (atomic_fp * reactive_f).sum(dim=0) / reactive_count
    global_pool = atomic_fp.mean(dim=0)

    reactive_fraction = reactive_f.mean()
    coord_stats = torch.stack(
        [
            coord_total.mean(),
            coord_total.std(unbiased=False),
            coord_total.max(),
            reactive_fraction,
        ]
    )

    anomaly_stats = torch.stack(
        [
            anomaly_scores.mean(),
            anomaly_scores.std(unbiased=False),
            anomaly_scores.max(),
        ]
    )

    frame_fp = torch.cat([reactive_pool, global_pool, coord_stats, anomaly_stats], dim=0)

    return anomaly_scores, atomic_fp, frame_fp, coord_total, reactive_mask


def extract_novel_frames_from_trajectory(
    traj_file,
    out_file,
    file_format="extxyz",
    top_k=20,
    candidate_multiplier=5,
    rc=3.5,
    z_surface_threshold=None,
    reactive_species=None,
    nms_tau=50,
    w_local=1.0,
    w_wasserstein=2.0,
    w_deviation=5.0,
    w_bond=3.0,
    mp_steps=3,
    mp_weight=0.5,
    bond_scale=1.25,
):
    """
    Two-stage selection:
      Stage 1: reactive-region novelty scoring + temporal NMS preselection
      Stage 2: FPS in frame embedding space
    """
    pool_size = top_k * candidate_multiplier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n[ Chemically Aware AL Sampler Initialized ]")
    print(f"-> Top K: {top_k}")
    print(f"-> Candidate pool size: {pool_size}")
    print(f"-> Message passing steps: {mp_steps}")
    print(f"-> Reactive-zone priority: {reactive_species if reactive_species is not None else 'surface mask only'}")

    traj_iter = iread(traj_file, index=":", format=file_format)
    try:
        ref_atoms = next(traj_iter)
    except StopIteration:
        print("Trajectory is empty.")
        return

    global_elements = np.unique(ref_atoms.numbers)
    N_ref = len(ref_atoms)

    candidate_heap = []
    history_frames = []
    history_scores = []

    last_selected_frame = torch.full((N_ref,), -9999, dtype=torch.long, device=device)

    # Reference frame
    ref_raw, ref_atomic_fp, ref_frame_fp, ref_coord_total, ref_reactive_mask = compute_atomic_features(
        ref_atoms,
        rc=rc,
        z_surface_threshold=z_surface_threshold,
        reactive_species=reactive_species,
        element_types=global_elements,
        device=device,
        mp_steps=mp_steps,
        mp_weight=mp_weight,
    )

    ref_score_field = ref_raw * ref_reactive_mask.float() if bool(ref_reactive_mask.any()) else ref_raw
    ref_signature = quantile_signature(ref_score_field[ref_reactive_mask] if bool(ref_reactive_mask.any()) else ref_score_field)

    history_frames.append(0)
    history_scores.append(
        {
            "total": 0.0,
            "local_anomaly": 0.0,
            "deviation": 0.0,
            "wasserstein": 0.0,
            "bond_change": 0.0,
        }
    )

    print("\n--- Phase 1: Pre-screening ---")
    for frame_idx, atoms in enumerate(traj_iter, start=1):
        if len(atoms) != N_ref:
            raise ValueError(
                f"Frame {frame_idx} has {len(atoms)} atoms, but reference has {N_ref}. "
                "Use a fixed-composition trajectory (NVT/NPT, no atom insertion/deletion)."
            )

        raw_scores, atomic_fp, frame_fp, coord_total, reactive_mask = compute_atomic_features(
            atoms,
            rc=rc,
            z_surface_threshold=z_surface_threshold,
            reactive_species=reactive_species,
            element_types=global_elements,
            device=device,
            mp_steps=mp_steps,
            mp_weight=mp_weight,
        )

        score_field = raw_scores * reactive_mask.float() if bool(reactive_mask.any()) else raw_scores

        # Reactive-region local anomaly
        if bool(reactive_mask.any()):
            local_max_anomaly = score_field[reactive_mask].max().item()
        else:
            local_max_anomaly = score_field.max().item()

        # Fingerprint deviation from reference
        atom_deviation = torch.norm(atomic_fp - ref_atomic_fp, dim=-1)
        if bool(reactive_mask.any()):
            max_deviation = atom_deviation[reactive_mask].max().item()
            bond_change = torch.abs(coord_total - ref_coord_total)[reactive_mask].mean().item()
        else:
            max_deviation = atom_deviation.max().item()
            bond_change = torch.abs(coord_total - ref_coord_total).mean().item()

        # Distribution shift in active zone
        current_signature = quantile_signature(score_field[reactive_mask] if bool(reactive_mask.any()) else score_field)
        wasserstein_dist = torch.mean(torch.abs(current_signature - ref_signature)).item()

        total_score = (
            w_local * local_max_anomaly
            + w_wasserstein * wasserstein_dist
            + w_deviation * max_deviation
            + w_bond * bond_change
        )

        history_frames.append(frame_idx)
        history_scores.append(
            {
                "total": total_score,
                "local_anomaly": local_max_anomaly,
                "deviation": max_deviation,
                "wasserstein": wasserstein_dist,
                "bond_change": bond_change,
            }
        )

        # Temporal non-maximum suppression on the active atom with maximum anomaly
        raw_for_nms = score_field.clone()
        max_atom_idx = int(torch.argmax(raw_for_nms).item())
        time_diff = frame_idx - last_selected_frame[max_atom_idx].item()
        penalty = 1.0 - np.exp(-max(time_diff, 0) / float(nms_tau))
        nms_score = local_max_anomaly * penalty

        item = (
            total_score,
            frame_idx,
            atoms.copy(),
            float(local_max_anomaly),
            float(max_deviation),
            float(wasserstein_dist),
            float(bond_change),
            frame_fp.detach().cpu(),
        )

        if len(candidate_heap) < pool_size:
            heapq.heappush(candidate_heap, item)
            last_selected_frame[max_atom_idx] = frame_idx
        else:
            if total_score > candidate_heap[0][0]:
                heapq.heappushpop(candidate_heap, item)
                last_selected_frame[max_atom_idx] = frame_idx

        if frame_idx % 50 == 0:
            print(
                f"Frame {frame_idx:05d} | score={total_score:.4f} | "
                f"local={local_max_anomaly:.4f} | dev={max_deviation:.4f} | "
                f"W1={wasserstein_dist:.4f} | bond={bond_change:.4f}"
            )

        if torch.cuda.is_available() and frame_idx % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    if len(candidate_heap) == 0:
        print("No candidates found.")
        return

    # Phase 2: FPS on frame-level chemical embeddings
    print("\n--- Phase 2: FPS on chemistry-aware frame embeddings ---")
    candidates = sorted(candidate_heap, key=lambda x: x[0], reverse=True)

    fps_mat = torch.stack([c[7] for c in candidates], dim=0).float()
    fps_mat = (fps_mat - fps_mat.mean(dim=0, keepdim=True)) / fps_mat.std(dim=0, keepdim=True).clamp_min(1e-6)

    num_to_select = min(top_k, len(candidates))
    selected = [0]
    min_dists = torch.cdist(fps_mat, fps_mat[[0]]).squeeze(1)

    for _ in range(1, num_to_select):
        min_dists[selected] = -1.0
        next_idx = int(torch.argmax(min_dists).item())
        selected.append(next_idx)
        d = torch.cdist(fps_mat, fps_mat[[next_idx]]).squeeze(1)
        min_dists = torch.minimum(min_dists, d)

    final_selection = [candidates[i] for i in selected]
    final_selection.sort(key=lambda x: x[1])

    novel_frames = []
    meta = []

    print(f"\n[ Done: {len(final_selection)} diverse frames selected ]")
    for rank, (score, f_idx, atoms_copy, loc, dev, wass, bond, _) in enumerate(final_selection, start=1):
        print(
            f"#{rank:02d} | Frame {f_idx:05d} | "
            f"score={score:.4f} | local={loc:.4f} | dev={dev:.4f} | W1={wass:.4f} | bond={bond:.4f}"
        )
        novel_frames.append(atoms_copy)
        meta.append(
            {
                "rank": rank,
                "frame_idx": int(f_idx),
                "score": float(score),
                "local_anomaly": float(loc),
                "deviation": float(dev),
                "wasserstein": float(wass),
                "bond_change": float(bond),
            }
        )

    write(out_file, novel_frames, format="extxyz")
    with open(os.path.splitext(out_file)[0] + "_selection_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nSaved selected frames to: {out_file}")
    print(f"Saved metadata to: {os.path.splitext(out_file)[0] + '_selection_meta.json'}")

    # Diagnostic plot
    if len(history_frames) > 1:
        totals = [s["total"] for s in history_scores]
        locs = [s["local_anomaly"] for s in history_scores]
        devs = [s["deviation"] for s in history_scores]
        wass = [s["wasserstein"] for s in history_scores]
        bonds = [s["bond_change"] for s in history_scores]

        plt.figure(figsize=(14, 7))
        plt.plot(history_frames, totals, label="Total novelty", linewidth=2.5)
        plt.plot(history_frames, locs, label="Local anomaly", linestyle="--", alpha=0.8)
        plt.plot(history_frames, devs, label="Fingerprint deviation", alpha=0.8)
        plt.plot(history_frames, wass, label="Wasserstein shift", alpha=0.8)
        plt.plot(history_frames, bonds, label="Bond-network change", alpha=0.8)

        selected_frames = [m["frame_idx"] for m in meta]
        selected_scores = [m["score"] for m in meta]
        plt.scatter(selected_frames, selected_scores, marker="*", s=220, label="FPS selected")

        plt.title("Chemistry-aware AL scoring timeline", fontsize=15)
        plt.xlabel("Frame index")
        plt.ylabel("Score")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plot_path = "al_scoring_timeline.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved diagnostic plot to: {plot_path}")


if __name__ == "__main__":
    traj_path = "traj.extxyz"

    if os.path.exists(traj_path):
        extract_novel_frames_from_trajectory(
            traj_file=traj_path,
            out_file="active_learning_candidates.xyz",
            file_format="extxyz",
            top_k=50,
            candidate_multiplier=5,
            rc=3.5,
            z_surface_threshold=0.0,   # replace with your actual substrate reference plane
            reactive_species=["Mo", "S", "O"],     # e.g. [1, 6, 7, 8, 16] or ["H", "C", "N", "O", "S"]
            nms_tau=80,
            w_local=1.0,
            w_wasserstein=0.2,
            w_deviation=10.0,
            w_bond=3.0,
            mp_steps=3,
            mp_weight=0.5,
            bond_scale=1.25,
        )
    else:
        print(f"Trajectory file not found: {traj_path}")
