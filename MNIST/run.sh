#!/bin/bash

epochs=20
nsamples_train=60000
momentum=0.0
p1=0.0
p2=0.0
lr=0.01
batch_size=6000
wtb=1
save=1


python wtb.py --wtb $wtb --nsamples_train $nsamples_train --batch_size $batch_size --lr=$lr --p1 $p1 --p2 $p2 --momentum $momentum --epochs $epochs --save $save


