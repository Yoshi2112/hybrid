#!/bin/bash
echo "Submitting all runs to grid"

PTH=CH4_tests
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run0.sh
sleep 10
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run1.sh
sleep 1
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run2.sh
sleep 1
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run3.sh
sleep 1
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run4.sh
sleep 1
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run5.sh
sleep 1
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run6.sh
echo "Done."
