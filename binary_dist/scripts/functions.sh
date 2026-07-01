### Useful functions

# Associative array for register_exit_trap_cmd function. 
# key -> value: priority -> command(s)
declare -A _trap_queue

# Accessory function to register_exit_trap_cmd
function _build_trap_cmds() {
    # We set the '_exit_status' variable at the beginning of the combined trapped commands to
    # capture the exit status of the EXIT signal.
    # This is needed because we cannot simply use `$?` directly within a command passed to
    # register_exit_trap_cmd because that would capture the exit status of the previous command
    # executed, which might be another command passed to register_exit_trap_cmd and not be the
    # original exit status of the script.
    # This way the '_exit_status' variable is available for all commands passed to 
    # register_exit_trap_cmd to refer to the original exit status of the script.
    local combined_cmds
    
    combined_cmds='_exit_status=$? ; '

    # Sort _trap_queue keys (priority) numerically and build the command chain in order of priority
    for priority in $(echo "${!_trap_queue[@]}" | tr ' ' '\n' | sort -n); do
        combined_cmds+="${_trap_queue[$priority]} ; "
    done

    trap "$combined_cmds" EXIT
}

function register_exit_trap_cmd() {
    # Register a command to be executed when the EXIT signal is triggered, with a priority level
    # that determines the order of execution relative to other registered commands.
    # Lower priority numbers run first, higher numbers run last.
    # Commands with the same priority run in registration order.
    # The commands registered are executed using `trap CMDs EXIT`, where CMDs is the combination
    # of all registered commands in the correct priority order.
    # 
    # Usage: register_exit_trap_cmd CMD [PRIORITY]
    #   CMD      - The command to run when the EXIT signal is triggered
    #   PRIORITY - Integer number defininig the execution order (default: 50). 
    #              Lower runs first, higher runs last.
    #
    # Note: to refer to the exit status of the script within trap commands, use
    # `$_exit_status` instead of `$?`, as `$?` will reflect the exit status of the
    # previously executed trap command rather than the original script exit status.
    #
    # Examples:
    #   register_exit_trap_cmd "my_fun"
    #   register_exit_trap_cmd "remove_fun" 90
    #   register_exit_trap_cmd "some_fun" 10
    #   register_exit_trap_cmd "my_other_fun"
    #
    #   will be executed as `trap 'some_fun ; my_fun ; my_other_fun ; remove_fun ; ' EXIT`
    local cmd priority
    
    cmd="$1"
    priority="${2:-50}"  # default priority 50, lower runs first
    
    # Add the command to the trap queue with the specified priority
    if [[ -v _trap_queue["$priority"] ]]; then
        _trap_queue["$priority"]+=" ; $cmd"
    else
        _trap_queue["$priority"]="$cmd"
    fi

    # Build the trap command based on the registered commands and their priorities
    _build_trap_cmds
}

function delete_files_in_manifest() {
    # Delete all files and folders associated with a version, which are listed in the manifest $1.
    local manifest_file="$1"
    if [[ ! -f "$manifest_file" ]]; then
        echo "Error: manifest file '$manifest_file' not found." >&2
        return 1
    fi
    # Make sure to split the manifest by newlines and not by spaces (-d '\n')
    # Do not run if the manifest is empty (--no-run-if-empty)
    xargs --no-run-if-empty -d '\n' rm -vrf < "$manifest_file"
}

function set_perms() {
    # Set permissions to a provided file or directory (recursively).
    # Use the -x option to set executable permissions to files.

    local OPTIND OPTARG opt default_exec_perm exec_perm arg acls
    
    # Default permission to capital X, to set executable permissions to files only
    # if any user already has executable permissions
    default_exec_perm='X'
    while getopts ":x" opt; do
        case $opt in
            x) exec_perm=x ;;
            \?) echo "Invalid option: -$OPTARG" >&2; return 1 ;;
        esac
    done
    shift $((OPTIND - 1))
    arg="$1"

    # Change group owner of files and directories recursively
    # we use the -h option to change symbolic links themselves and not the link they point to
    chgrp -R -h "$GROUP_OWNER" "$arg"

    # reset ACLs to make sure we don't have any non-wanted ACLs
    setfacl -R -b "$arg"

    ### Directories
    # Define permissions
    acls='u::rwx' # rwx for user
    acls+=',g::r-x' # r-x for group
    acls+=',o::---' # no permissions for others
    # Define default permissions
    # (newly created files/folders within the directory will inherit these permissions)
    acls+=',d:u::rwx' # rwx for user
    acls+=',d:g::r-x' # r-x for group
    acls+=',d:o::---' # no permissions for others
    # Set permissions and setgid (newly created files/folders will inherit the group owner of the directory)
    find "$arg" -type d \
        -exec setfacl -m "$acls" {} + \
        -exec chmod g+s {} +
    
    ### Files
    # Define permissions
    acls="u::rw${exec_perm:-$default_exec_perm}" # rwx for user if -x option is provided, otherwise rwX
    acls+=",g::r-${exec_perm:-$default_exec_perm}" # r-x for group if -x option is provided, otherwise r-X
    acls+=",o::---" # no permissions for others
    # Set permissions
    find "$arg" -type f \
        -exec setfacl -m "$acls" {} +
}