

old=ci_test_2026-09-17_09-10
new=ci_test_2026-09-17_11-44

cf_sandbox venv_multilepton_dev "python yield_table.py \
    --tests 22preEE_v12_central:data_mu_d \
    --version $old \
    --output yields_old.json"

cf_sandbox venv_multilepton_dev "python yield_table.py \
    --tests 22preEE_v12_central:data_mu_d \
    --version $new \
    --output yields_new.json"

python yield_table.py --compare yields_old.json yields_new.json
