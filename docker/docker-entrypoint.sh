#!/bin/bash
set -e

# Add the local bin directory to the PATH
export PATH="/home/user/.local/bin:$PATH"

# Decode COOKIES_B64 environment variable to /app/cookies.txt if present
# This allows passing the cookies file content via an environment variable (e.g. in Vast.ai)
if [ -n "$COOKIES_B64" ]; then
    echo "Found COOKIES_B64 environment variable. Decoding to /app/cookies.txt..."
    echo "$COOKIES_B64" | base64 -d > /app/cookies.txt
    echo "Successfully created /app/cookies.txt from environment variable."
fi

# Execute the command passed to the script
exec "$@"
