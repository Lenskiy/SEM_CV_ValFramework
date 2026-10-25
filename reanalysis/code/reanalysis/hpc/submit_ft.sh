#!/bin/bash
# Fine-tuning: 3 seeds x 5 folds of repeat 0. From .../zno_xai/code: bash submit_ft.sh
for s in 0 1 2; do for f in 0 1 2 3 4; do
  qsub -q adfagpu -N ft_s${s}_f${f} -o ../logs/ft_s${s}_f${f}.out -v FOLD=$f,SEED=$s finetune.pbs
done; done
