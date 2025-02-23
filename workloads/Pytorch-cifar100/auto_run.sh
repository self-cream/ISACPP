#!/bin/bash

: ${BATCH_SIZE:=128}
: ${NET:=''}
: ${EPOCHS:=1}

# Define the command you want to execute
command_to_run="python train.py -net ${NET} -gpu -b ${BATCH_SIZE} -epochs ${EPOCHS}"

# Define the interval in seconds (e.g., 10 seconds)
interval=1

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

