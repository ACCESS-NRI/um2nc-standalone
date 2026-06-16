# This script is sourced within the build_and_deploy.sh script,
# and it's used to build all files related to the module version.
# (e.g. modulefile, .modulerc, etc.)

# Create temporary jinja2 environment for rendering the modulefile template
env_prefix="./jinja2_env"
"$MAMBA_EXE" create \
    -y \
    --prefix $env_prefix \
    --no-rc \
    -c conda-forge \
    -c nodefaults \
    jinja2

# Render the modulefile template and save it to the module directory
"$MAMBA_EXE" run \
    --prefix $env_prefix \
    python "$INFRA_SCRIPTS_DIR/render_jinja2_template.py" \
    $INFRA_TEMPLATES_DIR/modulefile.j2 \
    > "$MODULE_FILE_PATH"

# Render the .modulerc template and save it to the module directory
"$MAMBA_EXE" run \
    --prefix $env_prefix \
    python "$INFRA_SCRIPTS_DIR/render_jinja2_template.py" \
    $INFRA_TEMPLATES_DIR/.modulerc.j2 \
    > "$MODULERC_FILE_PATH"

set_perms "$MODULE_FILE_PATH"
set_perms "$MODULERC_FILE_PATH"