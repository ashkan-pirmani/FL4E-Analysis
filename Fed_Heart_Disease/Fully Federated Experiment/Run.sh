#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/


sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 0 3`; do
    echo "Starting client $i"
    python clients.py --cid=${i} &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait