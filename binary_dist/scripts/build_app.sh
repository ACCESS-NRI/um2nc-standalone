# This script is sourced within the build_and_deploy.sh script, 
# and it's used to build all files inside the app version folder.
# (e.g. creating environment, creating binary executable, etc.)

# Creating pyinstaller environment
env_prefix="./pyinstaller_env"
if [[ "$ENV_TYPE" == STABLE ]]; then
    "$MAMBA_EXE" create \
        -y \
        --prefix $env_prefix \
        --no-rc \
        -c accessnri \
        -c conda-forge \
        -c nodefaults \
        pyinstaller \
        um2nc=="$MODULE_VERSION"
else
    "$MAMBA_EXE" create \
        -y \
        --prefix $env_prefix \
        --no-rc \
        -f "$REPO_PATH/.conda/env_dev.yml"
    # Install pyinstaller and um2nc
    "$MAMBA_EXE" run \
        --prefix $env_prefix \
        pip install \
            "$REPO_PATH" \
            pyinstaller \
            --no-build-isolation
fi

# Create binary executable with pyinstaller
dist_dir="./pyinstaller_dist"
"$MAMBA_EXE" run \
    --prefix $env_prefix \
    pyinstaller "$REPO_PATH/binary_dist/pyinstaller.spec" \
    --distpath "$dist_dir"

# Find the created executable
exe=$(find "$dist_dir" -type f -name "$MODULE_NAME"* -exec realpath {} \;)
exe_name=$(basename "$exe")

# Test the executable
version=$("$exe" --version)
echo "um2nc version: $version"
# Run integration tests
bash "$REPO_PATH/integration/regression_tests.sh" \
    --exe "$exe" \
    -q \
    -d full

# Move the created executable to the app version directory
mv -v "$exe" "$APP_VERSION_DIR"

export UM2NC_EXE="$APP_VERSION_DIR/$exe_name"

# Set um2nc executable permissions
set_perms -x "$UM2NC_EXE"

# Set symlink to the executable to um2nc for easier access
ln -sf "$UM2NC_EXE" "$APP_VERSION_DIR/um2nc"

# Copy executable to the TEMP_EXCHANGE_DIR for later use in GitHub Runner
cp -v "$UM2NC_EXE" "$TEMP_EXCHANGE_DIR"