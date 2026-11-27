#!/usr/bin/env bash
# Run the command lines given on stdin, at most JOBS at a time (default 3), each at
# nice 19. Used by the experiment scripts; not an experiment itself.
#   printf '%s\n' "cmd one" "cmd two" | old/scripts/par.sh
set -euo pipefail
grep -v '^[[:space:]]*$' | xargs -P "${JOBS:-3}" -d '\n' -I{} nice -n 19 sh -c '{}'
