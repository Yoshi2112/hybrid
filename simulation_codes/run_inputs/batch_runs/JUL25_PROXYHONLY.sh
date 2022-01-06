#!/bin/bash
NRUNS=16
echo "Submitting all runs to grid"

PTH1=JUL25_PROXYHONLY_5HE_PREDCORR
echo "Submitting 5PC runs..."
for value in `eval echo {0..$NRUNS}`
do
	qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH1/run${value}_${PTH1}.sh
	sleep 3
done

PTH2=JUL25_PROXYHONLY_30HE_PREDCORR
echo "Submitting 15PC runs..."
for value in `eval echo {0..$NRUNS}`
do
	qsub ~/hybrid/simulation_codes/run_inputs/batch_runs/$PTH2/run${value}_${PTH2}.sh
	sleep 3
done

echo "Done."
