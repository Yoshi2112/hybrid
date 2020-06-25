
An example of running matlab with multiple arguments

firstly change your main .m file to be a function with input parameters (matlabargs.m)

Then use the job script run-matlab-arcs.sh




You can then create a text file 'alljobs.txt' with hundreds of different parameters - one per line 


qsub -v VAR1=5,VAR2=7,OUTFILE=run1.mat run-matlab-args.sh
qsub -v VAR1=20,VAR2=30,OUTFILE=run2.mat run-matlab-args.sh
qsub -v VAR1=50,VAR2=37,OUTFILE=run3.mat run-matlab-args.sh
qsub -v VAR1=45,VAR2=37,OUTFILE=run4.mat run-matlab-args.sh


then do

rcglogin#   source alljobs.txt

and it will submit all of the jobs
