#!/bin/bash
#SBATCH --job-name=sd_clip_atrophy
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:59:00
#SBATCH --partition=biggpu
# No --exclusive (biggpu was heavily contested last time; non-exclusive
# co-location appears to be the norm on this partition -- see earlier
# notes). No --gres=gpu:1 -- this cluster's SLURM config doesn't track
# GPU as a Gres/TRES resource type even though a real GPU (RTX PRO 6000
# Blackwell) is physically present.

mkdir -p logs

# Uncomment and fill in if your interactive session needed a module load
# to get Python/CUDA working -- a submitted batch job starts with a clean
# environment and does not inherit anything loaded interactively.
# module load <exact name>

source venv/bin/activate

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# The streamlined pipeline is now 12 stages, orchestrated internally
# by run_full_pipeline.py itself -- no need to chain each stage manually
# here anymore (previously this script called each of 9 stages
# individually with its own echo/date wrapper). --skip_indexing assumes
# data/processed/metadata_processed.csv and image_index.pt already exist
# from a prior run; drop that flag if you want indexing rebuilt too.
echo "[$(date)] START run_full_pipeline.py"
python run_full_pipeline.py --skip_indexing
echo "[$(date)] ALL DONE"
