GeoRescue
GeoRescue is a training-free geometric point cloud registration framework designed for resource-constrained edge platforms.
This repository provides the main evaluation scripts, the ablation script, and the benchmark configuration files used for academic reproduction and validation in the paper.

Overview
The GeoRescue framework consists of three core modules:
ACE: Asymmetric Correspondence Expansion
DGTG: Dynamic Geometric Topology Gating
UAMR: Uncertainty-Aware Manifold Refinement
This repository currently includes:
Full-model evaluation script for 3DMatch
Full-model evaluation script for 3DLoMatch
Ablation script for 3DLoMatch
Benchmark configuration files (.pkl) used in the experiments

Repository Structure
GeoRescue/
├─ run_3dmatch.py
├─ run_3dlomatch.py
├─ run_ablation.py
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ benchmarks/
│ ├─ 3DMatch.pkl
│ ├─ 3DLoMatch.pkl
│ └─ ...
└─ outputs/

Environment
Recommended environment:
Python 3.10
Tested environment:
Windows 10
CPU-only evaluation
Open3D 0.18
Install dependencies with:
pip install -r requirements.txt

Requirements
Main third-party dependencies:
numpy
pandas
scipy
open3d
tabulate
tqdm

Data Preparation
This repository provides the benchmark configuration files (.pkl) required for reproduction, for example:
benchmarks/3DMatch.pkl
benchmarks/3DLoMatch.pkl
The full point cloud datasets should be downloaded from the official sources and organized according to the directory structure below.
Required Inputs
You need to prepare:
1.The point cloud root directory pcd_base_dir
2.The benchmark configuration file from the benchmarks/ folder
Expected Point Cloud Directory Structure
<pcd_base_dir>/
└─ <scene_name>/
├─ cloud_bin_0.ply
├─ cloud_bin_1.ply
├─ ...
Official Dataset Sources
3DMatch / 3DLoMatch: https://3dmatch.cs.princeton.edu/
KITTI Odometry: https://www.cvlibs.net/datasets/kitti/eval_odometry.php
Notes on Benchmark Files
The .pkl files in benchmarks/ specify the evaluation pairs used in this work.
They do not contain the full point cloud data. The original dataset files still need to be downloaded from the official sources listed above.

How to Run
1. Run Full Evaluation on 3DMatch
python run_3dmatch.py --pcd_base_dir "path/to/pointcloud_root" --benchmark_pkl "benchmarks/3DMatch.pkl" --output_dir "3dmatch_results"
Example:
python run_3dmatch.py --pcd_base_dir "E:\sanwei\3d\3d" --benchmark_pkl "benchmarks/3DMatch.pkl" --output_dir "3dmatch_results"
2. Run Full Evaluation on 3DLoMatch
python run_3dlomatch.py --pcd_base_dir "path/to/pointcloud_root" --benchmark_pkl "benchmarks/3DLoMatch.pkl" --output_dir "3dlomatch_results"
Example:
python run_3dlomatch.py --pcd_base_dir "E:\sanwei\3d\3d" --benchmark_pkl "benchmarks/3DLoMatch.pkl" --output_dir "3dlomatch_results"
3. Run the 3DLoMatch Ablation Study
python run_ablation.py --pcd_base_dir "path/to/pointcloud_root" --benchmark_pkl "benchmarks/3DLoMatch.pkl" --output_dir "ablation_results"
Example:
python run_ablation.py --pcd_base_dir "E:\sanwei\3d\3d" --benchmark_pkl "benchmarks/3DLoMatch.pkl" --output_dir "ablation_results"

Outputs
Outputs of run_3dmatch.py
GeoRescue_Full_details.csv
GeoRescue_Full_summary.csv
Outputs of run_3dlomatch.py
GeoRescue_Full_details.csv
GeoRescue_Full_summary.csv
Outputs of run_ablation.py
A_wo_ACE_details.csv
B_wo_DGTG_details.csv
C_wo_UAMR_details.csv
D_Full_details.csv
Final_Ablation_Table.csv
These files contain:
Per-sample detailed evaluation results
Full-model summary results
Final ablation summary tables

Correspondence to the Paper
This repository corresponds to the following experimental parts of the paper:
Main results on 3DMatch
Main results on 3DLoMatch
Ablation study on 3DLoMatch
Specifically:
run_3dmatch.py corresponds to the main 3DMatch results
run_3dlomatch.py corresponds to the main 3DLoMatch results
run_ablation.py corresponds to the ablation experiments reported in the paper

Notes
This repository is intended as a minimal public release for academic reproduction.
The current release focuses on 3DMatch / 3DLoMatch evaluation and ablation experiments.
The full official datasets are not redistributed in this repository and should be downloaded from their official sources.
Runtime and a small number of numerical results may vary slightly depending on the operating system, CPU model, Open3D version, and low-level implementation details.

Citation
If you use this code in your research, please cite the corresponding GeoRescue paper.
@article{georescue2026,
title={A Training-Free Geometric LiDAR Point Cloud Registration Method for Resource-Constrained Edge Platforms},
author={Sun, Yuyu and Shang, Zongkai and Yang, Mingxiao and Meng, Fandi and Mu, Mengxuan and Yan, Heqi},
journal={Sensors},
year={2026}
}

Contact
For academic questions regarding the code or experiments, please contact the corresponding author.