#!/bin/bash
PTH=JAN16_PKTS_5HE_CAMCL
echo "Submitting all runs to grid"
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run0_$PTH.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run1_$PTH.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run2_$PTH.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run3_$PTH.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run4_$PTH.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run5_$PTH.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run6_$PTH.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run7_$PTH.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run8_$PTH.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run9_$PTH.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run10_$PTH.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run11_$PTH.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH/run12_$PTH.sh
echo "Done."
