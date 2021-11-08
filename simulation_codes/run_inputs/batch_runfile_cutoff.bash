#!/bin/bash
PTH=PKT_START_5PERCENTHE_FROMCUTOFFS
echo "Submitting all runs to grid"
qsub ~/hybrid/simulation_codes/run_inputs/from_data/$PTH/run0_hybrid.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/from_data/$PTH/run1_hybrid.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/from_data/$PTH/run2_hybrid.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/from_data/$PTH/run3_hybrid.sh
sleep 5
qsub ~/hybrid/simulation_codes/run_inputs/from_data/$PTH/run4_hybrid.sh
echo "Done."
