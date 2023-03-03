source ~/mambaforge/etc/profile.d/conda.sh
conda activate topo
export TORCH_EXTENSIONS_DIR=~/torch_extensions/
export CUDA_HOME=~/mambaforge/envs/topo/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mambaforge/envs/topo/lib
export LIBRARY_PATH=$LIBRARY_PATH:~/mambaforge/envs/topo/lib/
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HYDRA_FULL_ERROR=1
python train.py "$@"
