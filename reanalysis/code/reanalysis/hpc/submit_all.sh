#!/bin/bash
# Submit the 49 full GPU runs. Run from /srv/scratch/$USER/zno_xai/code.
#   4  image-split reproduction (faithful), seed 0, folds 0-3
#   15 sensor CV image-only, seeds 0-2 x folds 0-4, with XAI
#   15 sensor CV + fabrication params, seeds 0-2 x folds 0-4
#   15 sensor CV including the outlier sensor, seeds 0-2 x folds 0-4
set -e
Q=adfagpu
sub() {  # name vars
  qsub -q $Q -N $1 -o ../logs/$1.out -v $2 train.pbs
}
for f in 0 1 2 3; do
  sub img_faith_f$f CV=image,FOLD=$f,FAITHFUL=1,SEED=0
done
for s in 0 1 2; do
  for f in 0 1 2 3 4; do
    sub sen_s${s}_f$f CV=sensor,FOLD=$f,SEED=$s,XAI=1
    sub senpar_s${s}_f$f CV=sensor,FOLD=$f,SEED=$s,PARAMS=1
    sub senout_s${s}_f$f CV=sensor,FOLD=$f,SEED=$s,OUTLIER=1
  done
done
