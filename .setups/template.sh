export cluster="manivald" # choices: "lxplus" or "manivald"
export CF_CERN_USER="$USER"
export CF_CERN_USER_FIRSTCHAR="${CF_CERN_USER:0:1}"
export CF_DATA="$CF_REPO_BASE/columnflow_venv"
export CF_SOFTWARE_BASE="$CF_DATA/software"
export CF_VENV_BASE="$CF_SOFTWARE_BASE/venvs"
export CF_STORE_NAME="cf_store"
export CF_WLCG_USE_CACHE="true"
export CF_WLCG_CACHE_CLEANUP="false"
export CF_WLCG_CACHE_MAX_SIZE=15GB
export CF_WLCG_CACHE_GLOBAL_LOCK="true"
export CF_VENV_SETUP_MODE_UPDATE="false"
export CF_VENV_SETUP_MODE="update"
export CF_INTERACTIVE_VENV_FILE=""
export CF_LOCAL_SCHEDULER="true"
export CF_SCHEDULER_HOST="127.0.0.1"
export CF_SCHEDULER_PORT="8082"
export CF_FLAVOR="cms"
export LAW_CMS_VO="cms"

if [ "$cluster" = "manivald" ]; then
    export OPENBLAS_NUM_THREADS=2
    export WLCG_FILE_SYSTEM="wlcg_fs_manivald"
    export CF_CRAB_STORAGE_ELEMENT="T2_EE_Estonia"
    export CF_CRAB_SANDBOX_NAME="CMSSW_10_6_18::arch=slc7_amd64_gcc700"
    export CF_SLURM_FLAVOR="manivald"
    export CF_SLURM_PARTITION="main"
    export CF_SLURM_CPUS=4
    export CF_SLURM_RUNTIME=6h
    export CF_SLURM_MEM_PER_CPU=8GB
    export CF_CLUSTER_LOCAL_PATH="/home/$CF_CERN_USER/HHMultilepton_Run3/"
    export TMPDIR="/scratch/local/$CF_CERN_USER"

elif [ "$cluster" = "lxplus" ]; then
    export WLCG_FILE_SYSTEM="wlcg_fs_cernbox"
    export CF_CRAB_STORAGE_ELEMENT="T2_CH_CERN"
    export CF_CRAB_SANDBOX_NAME="CMSSW_14_2_1::arch=el9_amd64_gcc21"
    export CF_HTCONDOR_FLAVOR="cern_el9"   # or "cern" for older versions of lxplus not using ELMA9
    export CF_HTCONDOR_MEMORY=2GB
    export CF_HTCONDOR_DISK=5GB
    export CF_HTCONDOR_RUNTIME=3h
    export CF_HTCONDOR_LOGS=true
    export CF_CLUSTER_LOCAL_PATH="/eos/user/$CF_CERN_USER_FIRSTCHAR/$CF_CERN_USER/HHMultilepton_Run3/"
    export TMPDIR="/tmp/$CF_CERN_USER"
fi

export CF_CRAB_BASE_DIRECTORY="/store/user/$CF_CERN_USER/HHMultilepton_Run3/cf_crab_outputs"
export CF_STORE_LOCAL="$CF_CLUSTER_LOCAL_PATH/$CF_STORE_NAME"
export CF_WLCG_CACHE_ROOT="$CF_CLUSTER_LOCAL_PATH/cf_scratch"
export CF_JOB_BASE="$CF_CLUSTER_LOCAL_PATH/cf_jobs"
