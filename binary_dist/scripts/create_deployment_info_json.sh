# This script produces a JSON file containing deployment information
# for a specific HPC target.
# The values for the JSON file fields are taken from the following env variables:
#
# - HPC_TARGET
# - MODULE_NAME
# - MODULE_VERSION
# - MODULE_TYPE
# - DEPLOYMENT_STAGE
# - STARTED_AT
# - MODULE_USAGE_INSTRUCTIONS
# - ENV_LOCK
# - SUCCESS
#
#
# Usage:
#   create_hpc_target_deployment_info_json.sh OUTPUT_JSON_FILE

set -euo pipefail
if [[ "${CONTAINERISED_ENVS_DEBUG:-0}" == "1" ]]; then
    set -x
fi

file="$1"

date=$(TZ='Australia/Sydney' date '+%FT%T %Z')
"$JQ_EXE" -n \
    --arg target "$HPC_TARGET" \
    --arg name "$MODULE_NAME" \
    --arg version "$MODULE_VERSION" \
    --arg type "$MODULE_TYPE" \
    --arg stage "$DEPLOYMENT_STAGE" \
    --arg started_at "$STARTED_AT" \
    --arg completed_at "$date" \
    --arg module_usage_instructions "${MODULE_USAGE_INSTRUCTIONS:-}" \
    --arg env_lock "${ENV_LOCK:-}" \
    --arg success "$SUCCESS" \
    '{
        "name": $target,
        "deployments": [
            {
                "module_name": $name,
                "module_version": $version,
                "module_type": $type,
                "deployment_stage": $stage,
                "started_at": $started_at,
                "completed_at": $completed_at,
                "module_usage_instructions": $module_usage_instructions,
                "env_lock": $env_lock,
                "success": $success
            }
        ]   
    }' > "$file"