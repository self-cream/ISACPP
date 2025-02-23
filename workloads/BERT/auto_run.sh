#!/bin/bash

: ${BATCH_SIZE:=1}
: ${EPOCHS:=1}

# Define the command you want to execute
command_to_run="bash scripts/run_squad.sh /workspace/bert/checkpoints/bert_large_qa.pt"

# Define the interval in seconds (e.g., 10 seconds)
interval=5

# Function to execute the command repeatedly
run_command() {
  while true; do
    # Execute the command
    $command_to_run

    # Sleep for the defined interval
    sleep $interval
  done
}

# Call the function to start running the command
run_command

