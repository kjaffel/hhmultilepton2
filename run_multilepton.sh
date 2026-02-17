#!/usr/bin/env bash

inherit=ml # "cf" or "ml"
task=${inherit}.GetDatasetLFNs
task=${inherit}.SelectEvents
task=${inherit}.ReduceEvents
task=${inherit}.PlotVariables1D
task=${inherit}.ProvideReducedEvents

#law run cf.GetDatasetLFNs --dataset data_mu_i --config 24_v15_central --remove-output 10
#law run ml.GetDatasetLFNs --dataset data_mu_i --config 24_v15_central --limit-dataset-files 1 --remove-output 10

law run ${task} \
    --version prod4 \
    --config 24_v15_central \
    --dataset qcd_mu_pt15to20_pythia \
    --workflow slurm --retries 1 --parallel-jobs 300 \
    ${1} 
    
    #--limit-dataset-files 1 \
    #--dataset dy_m50toinf_2j_pt600toinf_amcatnlo --limit-dataset-files 1 \
    # --branch 0 \
    # --dataset dy_m50toinf_2j_pt600toinf_amcatnlo \
    # --producers default \
    # --variables nmu \
    # --categories ceormu \
    # --view-cmd imgcat \
    # --remove-output 10 \
    # --workers 1 \
    # --print-status 2 \

    # FIXME to test out the functionality of these
    # --log-file slurm
    # --pilot 


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
