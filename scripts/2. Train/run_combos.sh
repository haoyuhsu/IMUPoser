#!/bin/bash

combos='global'

for combo in $combos
do 
  echo Running combo $combo
  # python 1.\ Train\ Global\ Model.py --combo_id $combo --experiment 'IMUPoserGlobalModel'  # --fast_dev_run
  # python 1.\ Train\ Global\ Model.py --combo_id $combo --experiment 'IMUPoserFineTuneDIP'  # --fast_dev_run

  # python 1.\ Train\ Global\ Model.py --combo_id $combo --experiment 'IMUPoserGlobalModel_humanml' --max_epochs 50 --dataset_name 'humanml'
  # python 1.\ Train\ Global\ Model.py --combo_id $combo --experiment 'IMUPoserFineTuneDIP_humanml' --max_epochs 20 --finetune \
  #   --pretrained_ckpt '../../checkpoints/IMUPoserGlobalModelThree_global-01292026-031253/epoch=epoch=18-val_loss=validation_step_loss=0.01020.ckpt'

  python 1.\ Train\ Global\ Model.py --combo_id $combo --experiment 'IMUPoserGlobalModel_lingo' --max_epochs 50 --dataset_name 'lingo'
  # python 1.\ Train\ Global\ Model.py --combo_id $combo --experiment 'IMUPoserFineTuneDIP_lingo' --max_epochs 20 --finetune \
  #   --pretrained_ckpt '../../checkpoints/IMUPoserGlobalModelThree_global-01292026-031253/epoch=epoch=18-val_loss=validation_step_loss=0.01020.ckpt'

done
