#!/bin/bash
PTH=JUL25_CLEANPEAK_MULTIPOP_FIXED
echo "Submitting CLEANPEAK_MULTIPOP runs..."
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run0.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run1.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run2.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run3.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run4.sh
echo "Done."
