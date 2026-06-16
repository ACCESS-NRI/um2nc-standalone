# This script is used to export variables from the GitHub runner environment to the HPC system

# Path to the infrastructure scripts
infra_scripts_dir="$REPO_PATH/binary_dist/scripts"
# Path to the infrastructure jinja2 templates
infra_templates_dir="$REPO_PATH/binary_dist/jinja2_templates"

# write export script
cat <<EOF
export TEMP_EXCHANGE_DIR='$TEMP_EXCHANGE_DIR'
export STABLE_PRODUCTION_BASE_DIR='$STABLE_PRODUCTION_BASE_DIR'
export REPO_PATH='$REPO_PATH'
export INFRA_SCRIPTS_DIR='$infra_scripts_dir'
export INFRA_TEMPLATES_DIR='$infra_templates_dir'
export MODULE_NAME='$MODULE_NAME'
export MODULE_VERSION='$MODULE_VERSION'
export GROUP_OWNER='$GROUP_OWNER'
export PBS_PROJECT='$PBS_PROJECT'
export PBS_STORAGE='$PBS_STORAGE'
export JQ_EXE='$JQ_EXE'
export STARTED_AT='$STARTED_AT'
export ENV_TYPE='$ENV_TYPE'
export HPC_TARGET='$HPC_TARGET'
export DEPLOYMENT_INFO_JSON_PATH='$DEPLOYMENT_INFO_JSON_PATH'
EOF
