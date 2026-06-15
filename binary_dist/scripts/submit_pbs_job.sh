#!/usr/bin/env bash

# This script was created based on the following script:
# https://git.nci.org.au/bom/ngm/conda-container/-/blob/ec3134fac7f0fd1fefc0a07ccdb532707f2ae222/submit.sh

# Used as `submit_pbs_job.sh job_script` to run job_script as a PBS job to NCI Gadi.

set -euo pipefail
set -x

if [ $# -ne 1 ]; then
  echo "Error: Exactly one argument (job_script) is required." >&2
  exit 1
fi

job_script="$1"

pbs_job_name="${JOB_NAME_PREFIX}_${MODULE_NAME}_${MODULE_VERSION}"

echo "Submitting PBS job '$job_script' using the following resource directives:"
echo "Name: '$pbs_job_name'"
echo "Project: '$PBS_PROJECT'"
echo "Storage: '$PBS_STORAGE'"

# The logs of a PBS job gets written only after the job completes. 
# Since we want to print the logs immediately as the processing progresses, we need
# to create a custom logfile where we send the STDOUT and STDERR of the PBS job to.
# For this to work, the PBS job script needs to redirect its process STDOUT and STDERR
# to this logfile (i.e. using `exec &> "$JOB_LOG_FILE"`)
export JOB_LOG_FILE=$(mktemp)
# We also create another logfile to send the default PBS job logs to
temp_pbs_log=$(mktemp)
# Both logfiles will be deleted when this script exits.
trap "rm -vf '$JOB_LOG_FILE' '$temp_pbs_log'" EXIT

# Using ::group:: to start GitHub Actions log grouping
echo "::group::=== JOB LOGS ==="
qsub \
  -N $pbs_job_name \
  -P "$PBS_PROJECT" \
  -l storage="${PBS_STORAGE}" \
  -m n \
  -V \
  -W block=true \
  -j oe \
  -o "$temp_pbs_log" \
  "$job_script" \
  &
# Get the PID of the qsub command, which is used to track when the PBS job finishes
QPID=$!

# Log STDOUT and STDERR of the PBS job log file in real-time
# stop when the QPID process ends
tail -F "$JOB_LOG_FILE" --pid=$QPID

# Get PBS job exit code
wait $QPID
qsub_exit_code=$?

# Using ::endgroup:: to end GitHub Actions log grouping
echo "::endgroup::"

exit $qsub_exit_code