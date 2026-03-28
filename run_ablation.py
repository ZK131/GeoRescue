# -*- coding: utf-8 -*-
"""
GeoRescue Ablation Runner (Public Release)

This script reproduces the 3DLoMatch ablation study for GeoRescue.
It evaluates four variants:
1. w/o ACE
2. w/o DGTG
3. w/o UAMR
4. Full model
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import pickle
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree


# ==============================================================================
# 1. Argument Parser
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run GeoRescue ablation experiments on 3DLoMatch.")
    parser.add_argument("--pcd_base_dir", type=str, required=True, help="Base directory of point cloud scenes.")
    parser.add_argument("--benchmark_pkl", type=str, required=True, help="Path to 3DLoMatch.pkl.")
    parser.add_argument("--output_dir", type=str, default="ablation_results", help="Directory to save outputs.")

    parser.add_argument("--sample_rate", type=int, default=1, help="Sampling rate for benchmark tasks.")
    parser.add_argument("--voxel_size", type=float, default=0.05, help="Voxel downsampling size.")
    parser.add_argument("--dist_thresh", type=float, default=0.075, help="RANSAC correspondence distance threshold.")
    parser.add_argument("--dgtg_tau", type=float, default=0.1, help="DGTG geometric consistency threshold.")

    parser.add_argument("--re_thresh", type=float, default=15.0, help="Rotation error threshold in degrees.")
    parser.add_argument("--te_thresh", type=float, default=30.0, help="Translation error threshold in cm.")

    parser.add_argument("--num_workers", type=int, default=max(1, multiprocessing.cpu_count() - 2),
                        help="Number of parallel workers.")
    return parser.parse_args()


ARGS = parse_args()

PCD_BASE_DIR = Path(ARGS.pcd_base_dir)
BENCHMARK_PKL = Path(ARGS.benchmark_pkl)
RESULT_SAVE_DIR = Path(ARGS.output_dir)

SAMPLE_RATE = ARGS.sample_rate
THRESHOLDS = {"RE": ARGS.re_thresh, "TE": ARGS.te_thresh}

CONFIG = {
    "voxel_size": ARGS.voxel_size,
    "dist_thresh": ARGS.dist_thresh,
    "dgtg_tau": ARGS.dgtg_tau,
}

VARIANTS = [
    {"id": "A_wo_ACE", "name": "w/o ACE", "reciprocal": True, "use_dgtg": True, "refine": True},
    {"id": "B_wo_DGTG", "name": "w/o DGTG", "reciprocal": False, "use_dgtg": False, "refine": True},
    {"id": "C_wo_UAMR", "name": "w/o UAMR", "reciprocal": False, "use_dgtg": True, "refine": False},
    {"id": "D_Full", "name": "GeoRescue (Full)", "reciprocal": False, "use_dgtg": True, "refine": True},
]

os.environ["OMP_NUM_THREADS"] = "1"
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
RESULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
NUM_WORKERS = max(1, ARGS.num_workers)


# ==============================================================================
# 2. Core Modules
# ==============================================================================
def preprocess(pcd):
    """Voxel downsampling + normal estimation + FPFH extraction."""
    pcd_down = pcd.voxel_down_sample(CONFIG["voxel_size"])
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def module_dgtg_soft_rank(src_pts, tgt_pts, corres):
    """Lightweight soft ranking for DGTG."""
    num_corr = len(corres)
    if num_corr < 50:
        return corres

    src_corr_pts = src_pts[corres[:, 0]]
    tgt_corr_pts = tgt_pts[corres[:, 1]]

    num_anchors = min(5, num_corr)
    anchor_idx = np.random.choice(num_corr, num_anchors, replace=False)

    votes = np.zeros(num_corr)
    for idx in anchor_idx:
        dist_src = np.linalg.norm(src_corr_pts - src_corr_pts[idx], axis=1)
        dist_tgt = np.linalg.norm(tgt_corr_pts - tgt_corr_pts[idx], axis=1)
        votes += (np.abs(dist_src - dist_tgt) < (CONFIG["dgtg_tau"] * 2)).astype(int)

    sorted_indices = np.argsort(-votes)
    return corres[sorted_indices]


def module_refine_uamr(src_down, tgt_down, init_transform):
    """UAMR refinement approximated by point-to-plane ICP."""
    try:
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        reg = o3d.pipelines.registration.registration_icp(
            src_down,
            tgt_down,
            CONFIG["voxel_size"] * 2,
            init_transform,
            estimator,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
        )
        return reg.transformation
    except Exception:
        return init_transform


def run_pipeline(src_pcd, tgt_pcd, variant_cfg):
    """Run the full or ablated pipeline."""
    start_time = time.time()

    src_down, src_fpfh = preprocess(src_pcd)
    tgt_down, tgt_fpfh = preprocess(tgt_pcd)

    # 1. Matching
    src_feat = src_fpfh.data.T
    tgt_feat = tgt_fpfh.data.T

    if variant_cfg["reciprocal"]:
        tree_src = cKDTree(tgt_feat)
        _, idx_src = tree_src.query(src_feat, k=1)

        tree_tgt = cKDTree(src_feat)
        _, idx_tgt = tree_tgt.query(tgt_feat, k=1)

        corres = np.array([[i, m] for i, m in enumerate(idx_src) if idx_tgt[m] == i])
    else:
        tree = cKDTree(tgt_feat)
        _, idx = tree.query(src_feat, k=1)
        corres = np.stack((np.arange(len(idx)), idx), axis=1)

    if len(corres) < 3:
        return np.eye(4), time.time() - start_time

    coarse_transform = np.eye(4)
    best_fitness = -1.0

    # 2. Coarse estimation
    if variant_cfg["use_dgtg"]:
        src_np = np.asarray(src_down.points)
        tgt_np = np.asarray(tgt_down.points)
        ranked_corr = module_dgtg_soft_rank(src_np, tgt_np, corres)

        elite_ratio = 0.3
        elite_size = max(50, int(len(ranked_corr) * elite_ratio))
        elite_corr = ranked_corr[:elite_size]

        try:
            reg_elite = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                src_down,
                tgt_down,
                o3d.utility.Vector2iVector(elite_corr),
                CONFIG["dist_thresh"],
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3,
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(CONFIG["dist_thresh"])],
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
            )
            if reg_elite.fitness > 0.1:
                coarse_transform = reg_elite.transformation
                best_fitness = reg_elite.fitness
        except Exception:
            pass

        if best_fitness <= 0.1:
            try:
                reg_full = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                    src_down,
                    tgt_down,
                    o3d.utility.Vector2iVector(ranked_corr),
                    CONFIG["dist_thresh"],
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    3,
                    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(CONFIG["dist_thresh"])],
                    o3d.pipelines.registration.RANSACConvergenceCriteria(2000000, 0.999),
                )
                coarse_transform = reg_full.transformation
            except Exception:
                pass
    else:
        try:
            reg_naive = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                src_down,
                tgt_down,
                o3d.utility.Vector2iVector(corres),
                CONFIG["dist_thresh"],
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3,
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(CONFIG["dist_thresh"])],
                o3d.pipelines.registration.RANSACConvergenceCriteria(2000000, 0.999),
            )
            coarse_transform = reg_naive.transformation
        except Exception:
            pass

    # 3. Refinement
    if variant_cfg["refine"]:
        final_transform = module_refine_uamr(src_down, tgt_down, coarse_transform)
    else:
        final_transform = coarse_transform

    return final_transform, time.time() - start_time


def check_metrics(pred_transform, gt_transform):
    """Check success and compute TE/RE."""
    try:
        diff = np.linalg.inv(gt_transform) @ pred_transform
        te = np.linalg.norm(diff[:3, 3]) * 100.0
        re = np.rad2deg(np.arccos(np.clip((np.trace(diff[:3, :3]) - 1) / 2, -1, 1)))
        success = (te < THRESHOLDS["TE"]) and (re < THRESHOLDS["RE"])
        return success, te, re
    except Exception:
        return False, 999.0, 999.0


def worker(args):
    """Single-pair evaluation worker."""
    info, frag_id0, frag_id1, gt_transform, variant_cfg = args
    try:
        src_path = PCD_BASE_DIR / info["scene_name"] / f"cloud_bin_{frag_id1}.ply"
        tgt_path = PCD_BASE_DIR / info["scene_name"] / f"cloud_bin_{frag_id0}.ply"

        src_pcd = o3d.io.read_point_cloud(str(src_path))
        tgt_pcd = o3d.io.read_point_cloud(str(tgt_path))

        if src_pcd.is_empty() or tgt_pcd.is_empty():
            return None

        pred_transform, runtime_cost = run_pipeline(src_pcd, tgt_pcd, variant_cfg)
        success, te, re = check_metrics(pred_transform, gt_transform)

        return {
            "succ": success,
            "te": te,
            "re": re,
            "time": runtime_cost,
        }
    except Exception:
        return None


# ==============================================================================
# 3. Main
# ==============================================================================
def main():
    print("=" * 80)
    print("GeoRescue Ablation Study on 3DLoMatch")
    print("=" * 80)

    if not BENCHMARK_PKL.exists():
        print(f"[ERROR] Benchmark file not found: {BENCHMARK_PKL}")
        return

    with open(BENCHMARK_PKL, "rb") as f:
        tasks = pickle.load(f)[::SAMPLE_RATE]

    total_tasks = len(tasks)
    print(f"[INFO] Loaded {total_tasks} benchmark pairs.\n")

    summary = []

    for variant in VARIANTS:
        print(f"[RUN] Variant: {variant['name']}")

        job_args = []
        for item in tasks:
            gt_transform = np.vstack(
                (
                    np.hstack((item["rotation"], item["translation"].reshape(3, 1))),
                    [0, 0, 0, 1],
                )
            )
            job_args.append((item, item["frag_id0"], item["frag_id1"], gt_transform, variant))

        with multiprocessing.Pool(NUM_WORKERS) as pool:
            results = []
            for idx, ret in enumerate(pool.imap_unordered(worker, job_args)):
                if ret is not None:
                    results.append(ret)
                if (idx + 1) % 50 == 0:
                    print(f"\r   Progress: {idx + 1}/{total_tasks}", end="", flush=True)
        print("\n")

        detail_df = pd.DataFrame(results)
        detail_df.to_csv(RESULT_SAVE_DIR / f"{variant['id']}_details.csv", index=False)

        rr = (detail_df["succ"].sum() / total_tasks) * 100 if total_tasks > 0 else 0.0
        succ_df = detail_df[detail_df["succ"] == True]
        te = succ_df["te"].mean() if len(succ_df) > 0 else 999.0
        re = succ_df["re"].mean() if len(succ_df) > 0 else 999.0
        tm = detail_df["time"].mean() if len(detail_df) > 0 else 999.0

        print(f"   RR={rr:.2f}% | TE={te:.2f} cm | RE={re:.2f}° | Time={tm:.3f}s\n")

        summary.append(
            {
                "Variant": variant["name"],
                "RR (%)": rr,
                "TE (cm)": te,
                "RE (°)": re,
                "Time (s)": tm,
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULT_SAVE_DIR / "Final_Ablation_Table.csv", index=False)

    print("=" * 80)
    print("Final 3DLoMatch Ablation Table")
    print("=" * 80)
    try:
        print(summary_df.to_markdown(index=False, floatfmt=".2f"))
    except Exception:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
