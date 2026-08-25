
# source this file before running PPD calculations
# It'll set up the necessary environment which includes tensorflow
# set up to use GPUS

module load conda
#conda activate /global/common/software/des/aferte/myperltf
# slightly updated version to work with newer cudatoolkit
conda activate  /global/common/software/des/jmuir/myperltf_march2025

# Note that previously when CUDA tools on NERSC were updated,
# this stopoed working. If this happens again we may need to
# do some experimenting about version of these modules and
# the conda environment set up.

module load cudatoolkit/12.4
module load cudnn/8.9.3_cuda12

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME
#export TF_GPU_ALLOCATOR=cuda_malloc_async # tried after oom error, didn't help
#needed for plots
module load texlive
