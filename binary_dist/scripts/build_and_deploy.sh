#!/usr/bin/env bash

#PBS -q copyq
#PBS -l ncpus=1
#PBS -l walltime=1:00:00
#PBS -l mem=20GB
#PBS -l jobfs=10GB
#PBS -W umask=0037

set -euo pipefail
set -x

# Redirect STDOUT and STDERR of this shell to the log file, to 
# be able to capture all STDOUT and STDERR of the scheduled job without having to wait
# for the job to end
exec &> "$JOB_LOG_FILE"

# Set configuration env variables
source "$INFRA_SCRIPTS_DIR/install_config.sh"

# Change to the temporary working directory
cd "$TEMP_WORKING_DIR"

# Make sure the target app version directory does not already exist, to avoid accidentally overwriting an existing version.
if [[ -d "$APP_VERSION_DIR" ]]; then
    echo "Error! App version '$MODULE_NAME/$MODULE_VERSION' already exists." >&2
    exit 1
fi

### Initialise directories
# Create a trap function that would delete the app version related files
# in case the script fails
cleanup_env() {
    # _exit_status vaiable is initialised within the register_exit_trap_cmd function
    if [ $_exit_status -ne 0 ]; then
        echo "Error! Build failed. Cleaning up version '$MODULE_NAME/$MODULE_VERSION' related files..." >&2
        delete_files_in_manifest "$FILES_MANIFEST_PATH"
    fi
}
register_exit_trap_cmd cleanup_env $TRAP_PRIORITY_LAST

echo 'Initialising directories...'
if [[ ! -d "$BASE_DIR" ]]; then
    mkdir -p "$BASE_DIR"
    set_perms "$BASE_DIR"
fi
mkdir -pv "$APP_VERSION_DIR"
mkdir -pv "$MODULE_DIR"

### Create files_manifest file
# Create a files_manifest file, listing all the files and folders related to the current version deployment:
# - Modulefile
# - App version folder

cat > "$FILES_MANIFEST_PATH" <<EOF
$MODULE_FILE_PATH
$APP_VERSION_DIR
EOF

# # Set a trap function to update the HPC target deployment info JSON when the script exits
# update_hpc_target_deployment_info() {
#     # _exit_status variable is initialised within the register_exit_trap_cmd function
#     if [ $_exit_status -eq 0 ]; then
#         export SUCCESS=true
#         # Module usage instructions
#         export MODULE_USAGE_INSTRUCTIONS="module use $ALL_MODULES_DIR\nmodule load $MODULE_NAME/$MODULE_VERSION"
#     else
#         export SUCCESS=false
#     fi
#     # Update the HPC target deployment info JSON
#     source "$INFRA_SCRIPTS_DIR/create_hpc_target_deployment_info_json.sh" "$HPC_TARGET_DEPLOYMENT_INFO_JSON_PATH"
# }
# register_exit_trap_cmd update_hpc_target_deployment_info $TRAP_PRIORITY_FIRST

### Build App
source "$INFRA_SCRIPTS_DIR/build_app.sh"

### Deploy module files
source "$INFRA_SCRIPTS_DIR/build_module.sh"

### Cleanup oldest development app versions
if [[ "$ENV_TYPE" == DEVELOPMENT ]]; then
    source "$INFRA_SCRIPTS_DIR/cleanup_old_dev_versions.sh"
fi

### Ensure right permissions recursively for the app version
set_perms "$APP_VERSION_DIR"