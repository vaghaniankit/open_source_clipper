#!/bin/bash
set -e

# Add the local bin directory to the PATH
export PATH="/home/user/.local/bin:$PATH"

# Execute the command passed to the script
exec "$@"
