#!/bin/bash

NUM_CLIENTS=3  # Number of clients

echo "Starting server"
python server.py &
SERVER_PID=$!  # Get the server process ID
sleep 3  # Sleep for 3s to give the server enough time to start

# Start clients in the background
for i in $(seq 1 "$NUM_CLIENTS"); do
    echo "Starting client $i"
    python client.py 2> /dev/null &
done

# This will allow you to use CTRL+C to stop all background processes
trap "kill $SERVER_PID && pkill -P $$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
