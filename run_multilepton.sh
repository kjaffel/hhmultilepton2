#!/usr/bin/env bash

task=cf.ReduceEvents
task=cf.PlotVariables1D
task=cf.ProvideReducedEvents
task=cf.GetDatasetLFNs
task=cf.SelectEvents

requested_datasets=(
 data_mu_e
 qcd_mu_pt20to30_pythia
 wmh_wqq_hbb_powheg
)
requested_datasets_not_now=(
 qcd_mu_pt30to50_pythia
 qcd_mu_pt50to80_pythia
 qcd_mu_pt80to120_pythia
 qcd_mu_pt120to170_pythia
 qcd_mu_pt170to300_pythia
 qcd_mu_pt300to470_pythia
 qcd_mu_pt470to600_pythia
 qcd_mu_pt600to800_pythia
 qcd_mu_pt800to1000_pythia
 qcd_mu_pt1000toinf_pythia
)

for dataset in ${requested_datasets[*]}; do 
    law run ${task} \
        --version fixbranches_divergence_from_master_1 \
        --config 24_v15_central \
        --dataset $dataset \
        --retries 1 \
        --clear-logs \
        --cleanup-jobs \
        --limit-dataset-files 1 \
        ${1} 
done
    
    # --parallel-jobs 300 \
    # --workflow slurm \
    # --branch 0 \
    # --producers default \
    # --variables nmu \
    # --categories ceormu \
    # --view-cmd imgcat \
    # --remove-output 10 \
    # --workers 1 \

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
