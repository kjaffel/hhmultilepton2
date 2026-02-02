#!/usr/bin/env bash

#debug="LAW_LOG_LEVEL=DEBUG LAW_DEBUG=1 "
debug=""

#task="cf.SelectEvents"
task="ml.PlotVariables1D"
#task="ml.GetDatasetLFNs"

#law run cf.GetDatasetLFNs --dataset data_mu_i --config 24_v15_central --remove-output 10
#law run ml.GetDatasetLFNs --dataset data_mu_i --config 24_v15_central --limit-dataset-files 1 --remove-output 10

${debug}law run ${task} \
    --version onefile_test \
    --configs 24_v15_central \
    --datasets data_mu_i \
    --producers default \
    --variables nmu \
    --categories ceormu \
    --view-cmd imgcat \
    --limit-dataset-files 1 \
    ${1} 
    
    #--remove-output 10 \
    # --workflow slurm \
    # --workers 1 \
    # --branch 0 \
    # --parallel-jobs 300 \
    # --print-status 2 \

    # FIXME to test out the functionality of these
    # --log-file slurm
    # --pilot 

# remove outputs (example):
#   law run cf.SelectEvents --version mytest --remove-output  3

# options: 
#   --configs: 
#        22preEE_v14_private, 22postEE_v14_private, 23preBPix_v14_private, 23postBPix_v14_private
#        22preEE_v12_central, 22postEE_v12_central, 23preBPix_v12_central, 23postBPix_v12_central, 24_v15_central
#   --processes:
#       all_data, all_signals, all_backgrounds,       
#       resonant, nonresonant, nonresonant_ggf, nonresonant_vbf
#       ggf_4v, ggf_4t, ggf_2t2v, vbf_4v, vbf_4t, vbf_2t2v
#       4v, 4t, 2t2v
#   --datasets:
#       all_data, all_backgrounds, all_signals
#       ttbar, single_top, dy, wjets, qcd, zz, single_higgs, vvv, others
