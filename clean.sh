#!/bin/sh

timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p logs
mv srun.out srun.err log.txt "logs/${timestamp}_"
