#!/bin/bash
# One-time environment setup on the Katana LOGIN node (compute nodes may lack
# internet). Everything lives on scratch: the venv and the pip cache.
#   nohup setsid bash setup_env.sh > ../logs/setup_env.log 2>&1 < /dev/null &
set -e
ENV=/srv/scratch/$USER/envs/zno_xai
export PIP_CACHE_DIR=/srv/scratch/$USER/pip_cache

module load python/3.11.3
python3 -m venv "$ENV"
source "$ENV/bin/activate"
pip install --upgrade pip wheel

# CUDA 12.1 build; L40S nodes (Ada) are supported.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install torchcam==0.4.0 numpy scipy scikit-learn pillow matplotlib tqdm

python -c "import torch, torchvision, torchcam, sklearn, scipy, PIL; print('torch', torch.__version__, 'cuda build', torch.version.cuda, '| torchcam', torchcam.__version__, '| sklearn', sklearn.__version__)"
pip freeze > "$ENV/requirements.lock.txt"
echo "RESULT setup_env done"
