# This file is sourced in the build_and_deploy.sh script to cleanup oldest DEVELOPMENT
# versions if the number of dev versions is greater than MAX_DEV_VERSIONS

app_versions=$(
    find "$APP_VERSION_DIR/.." \
    -mindepth 1 -maxdepth 1 \
    -type d \
    -printf '%T+ %p\n' \
    | sort
)
num_versions=$(echo "$app_versions" | wc -l)
if [[ $num_versions -gt $MAX_DEV_VERSIONS ]]; then
    oldest_version_dir=$(echo "$app_versions" | head -n 1 | cut -d' ' -f2-)
    oldest_version_manifest="$oldest_version_dir/$FILES_MANIFEST_NAME"
    oldest_version=$(basename "$oldest_version_dir")
    msg="Number of '$MODULE_NAME' DEVELOPMENT versions in PRODUCTION greater than '$MAX_DEV_VERSIONS'. "
    msg+="Cleaning up oldest '$MODULE_NAME' DEVELOPMENT version: $oldest_version"
    echo "$msg"
    delete_files_in_manifest "$oldest_version_manifest"
fi