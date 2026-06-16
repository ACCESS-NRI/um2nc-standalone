# Sourced by 'binary_dist/build_and_deploy.sh' to configure the deployment of the module.

# Make functions available
source "$INFRA_SCRIPTS_DIR/functions.sh"

# Set named constant priorities for the register_exit_trap_cmd function
export TRAP_PRIORITY_FIRST=10 # Runs first (Used for setup commands)
export TRAP_PRIORITY_LAST=90 # Runs last (e.g. used for commands that delete files/folders)

# Maximum number of DEVELOPMENT versions to keep simultaneously.
# If a new deployment causes the total to exceed this limit, the oldest version is deleted.
export MAX_DEV_VERSIONS=3

# Sanity check on ENV_TYPE
if [[ "$ENV_TYPE" != STABLE && "$ENV_TYPE" != DEVELOPMENT ]]; then
    echo "Error: Invalid ENV_TYPE '$ENV_TYPE'. Must be either 'STABLE' or 'DEVELOPMENT'." >&2
    exit 1
fi

# Set BASE_DIR depending on the environment type:
if [[ "$ENV_TYPE" == DEVELOPMENT ]]; then
    BASE_DIR="$STABLE_PRODUCTION_BASE_DIR/prerelease"
else
    BASE_DIR="$STABLE_PRODUCTION_BASE_DIR"
fi
export BASE_DIR

# Path to the directory containing all versions of the app
app_dir="$BASE_DIR/apps/$MODULE_NAME"
# Path to the directory containing the specific version of the app
export APP_VERSION_DIR="$app_dir/$MODULE_VERSION"
# Files manifest for tracking all files for a specific app version
export FILES_MANIFEST_NAME=files_manifest.txt
export FILES_MANIFEST_PATH="$APP_VERSION_DIR/$FILES_MANIFEST_NAME"
# Path to the directory containing all versions of the module
export MODULE_DIR="$BASE_DIR/modules/$MODULE_NAME"
# Full path of the modulefile
export MODULE_FILE_PATH="$MODULE_DIR/$MODULE_VERSION"
# Full path of the modulefile
export MODULERC_FILE_PATH="$MODULE_DIR/.modulerc"

# Create temporary working directory
export TEMP_WORKING_DIR="$(mktemp -d)"
# Delete the temporary working directory at exit
register_exit_trap_cmd "rm -rfv $TEMP_WORKING_DIR" $TRAP_PRIORITY_LAST

### Micromamba initialisation
export MAMBA_EXE="${MAMBA_EXE:-}"
# Set MAMBA_ROOT_PREFIX to a temporary directory within the TEMP_WORKING_DIR
# MAMBA_ROOT_PREFIX in our case only controls conda packages caching
export MAMBA_ROOT_PREFIX="$TEMP_WORKING_DIR/micromamba_root"
mkdir -pv "$MAMBA_ROOT_PREFIX"
# If MAMBA_EXE is not found or not executable, a temporary micromamba executable is installed
if [ ! -x "$MAMBA_EXE" ]; then
    echo "Micromamba executable '$MAMBA_EXE' not found or not executable."
    echo "Installing micromamba's latest version:"
    mamba_temp_exe="$MAMBA_ROOT_PREFIX/micromamba"
    mamba_download_url='https://micro.mamba.pm/api/micromamba/linux-64/latest'
    curl -L "$mamba_download_url" | tar -xvjO bin/micromamba > "$mamba_temp_exe"
    # Set executable permissions to the micromamba executable
    set_perms -x "$mamba_temp_exe"
    MAMBA_EXE="$mamba_temp_exe"
fi
echo "Using micromamba executable: $MAMBA_EXE"
echo "Micromamba version: $($MAMBA_EXE --version)"

### jq initialisation
export JQ_EXE="${JQ_EXE:-}"
# If the jq executable is not found or not executable, a temporary jq executable is installed
if [ ! -x "$JQ_EXE" ]; then
    echo "jq executable '$JQ_EXE' not found or not executable."
    echo "Installing jq's latest version:"
    JQ_EXE="$TEMP_WORKING_DIR/jq"
    jq_download_url='https://github.com/jqlang/jq/releases/latest/download/jq-linux-amd64'
    curl -L "$jq_download_url" --output "$JQ_EXE"
    # Set executable permissions to the jq executable
    set_perms -x "$JQ_EXE"
fi
echo "Using jq executable: $JQ_EXE"
echo "jq version: $($JQ_EXE --version)"