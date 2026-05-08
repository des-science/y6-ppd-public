
# source this file before running PPD calculations
# It'll set up the necessary environment which includes tensorflow
# set up to use GPUS

# You can also use your own tensorflow installation, but getting that
#  working can be tricky and can result in gigantic conda envs if
#  you're not careful. This env is adapted from one initially set up
#  by Agnes Ferte.

# It may need to be updated further if perlmutter updates change
# the available CUDA versions.

module load conda

#conda activate /global/common/software/des/aferte/myperltf
# updated version to work with newer cudatoolkit
conda activate  /global/common/software/des/jmuir/myperltf_march2025

module load cudatoolkit/12.4
module load cudnn/8.9.3_cuda12

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

#needed for plots
module load texlive
