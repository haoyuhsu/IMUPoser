# IMUPoser: Full-Body Pose Estimation using IMUs in Phones, Watches, and Earbuds

> **IMU4D fork.** This copy is rewritten around SMPL-X (22 body joints) and streams IMU4D WebDataset
> shards at runtime instead of preprocessing AMASS / DIP-IMU. See sections 3-4 and "Code structure".
Click to watch the video!
<p align="center">
  <a href="https://www.youtube.com/watch?v=hgpjbKv8XFY"><img src="media/IMUPoser_github.png" alt="animated" width="100%"/></a>
</p>

Research code for IMUPoser (CHI 2023)

## Reference
Vimal Mollyn, Riku Arakawa, Mayank Goel, Chris Harrison, and Karan Ahuja. 2023. IMUPoser: Full-Body Pose Estimation using IMUs in Phones, Watches, and Earbuds. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems (CHI '23). Association for Computing Machinery, New York, NY, USA, Article 529, 1–12.

[Paper](https://dl.acm.org/doi/10.1145/3544548.3581392) | [Website](https://www.figlab.com/research/2023/imuposer)


BibTeX Reference:

```
@inproceedings{10.1145/3544548.3581392,
author = {Mollyn, Vimal and Arakawa, Riku and Goel, Mayank and Harrison, Chris and Ahuja, Karan},
title = {IMUPoser: Full-Body Pose Estimation Using IMUs in Phones, Watches, and Earbuds},
year = {2023},
isbn = {9781450394215},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3544548.3581392},
doi = {10.1145/3544548.3581392},
booktitle = {Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems},
articleno = {529},
numpages = {12},
keywords = {sensors, inertial measurement units, mobile devices, Motion capture},
location = {Hamburg, Germany},
series = {CHI '23}
}
```

## 1. Clone (or Fork!) this repository
```
git clone https://github.com/FIGLAB/IMUPoser.git
```
 
## 2. Create a virtual environment
See `install.sh` (conda, `python 3.8`, torch 2.x, `pip install -e src/`). The IMU4D checkout is
used read-only for `imu_synthesis` and the data; see section 3 for the environment variables.

## 3. Data: IMU4D SMPL-X shards (no preprocessing)

This fork trains directly on the IMU4D WebDataset shards under
`$IMU4D_DATA_ROOT/processed/<dataset>/<version>/wds/{train,val,test}/*.tar`. Nothing is converted
or cached: every clip is read from the tar stream at runtime and turned into model inputs on the
fly (IMU synthesis, first-frame alignment, SMPL-X kinematics). Locations are resolved by
`src/imuposer/imu4d/paths.py`:

| Variable | Meaning | Default |
|---|---|---|
| `IMU4D_ROOT` | IMU4D checkout providing `imu_synthesis/` | the checkout that contains this `baseline/IMUPoser` |
| `IMU4D_DATA_ROOT` | data tree with `processed/` | `$IMU4D_ROOT/data` |
| `SMPLX_MODEL_PATH` | `SMPLX_NEUTRAL.npz` | `$IMU4D_DATA_ROOT/models/smplx/SMPLX_NEUTRAL.npz`, then `../body_models/human_model_files/smplx/SMPLX_NEUTRAL.npz` |

Datasets are named by their spec, e.g. `humoto/v1`, `hiphi/v1`, `motionmillion/v1` (virtual
IMUs from `imu_traj`) or `imuposer/v2`, `dipimu/v2`, `ncsa/v1` (measured IMUs). Several specs can
be mixed with commas.

## 4. Training and evaluation

```bash
# pretrain (wandb logger by default; --logger csv needs no account)
python scripts/train.py --experiment imuposer_mm --datasets motionmillion/v1,hiphi/v1 --max_epochs 50
# fine-tune on real IMUs from a checkpoint
python scripts/train.py --experiment imuposer_ft --datasets imuposer/v2 \
    --pretrained_ckpt checkpoints/<run>/<ckpt>.ckpt --lr 5e-5 --max_epochs 30 --batch_size 32
# export predictions (one .npz per test sequence) and score them with the IMU4D metric
python scripts/evaluate.py --checkpoint checkpoints/<run>/<ckpt>.ckpt --datasets humoto/v1 --combo lw_rp_h
python -m metric.motion --input-dir baseline/IMUPoser/predictions/humoto_v1/lw_rp_h --max-frames 60   # from IMU4D root
```

`scripts/run_train.sh`, `scripts/run_finetune_real.sh` and `scripts/run_eval.sh` wrap these with
environment-variable overrides (`DATASETS`, `EXPERIMENT`, `PRETRAINED`, `CHECKPOINT`, `COMBOS`).

## Code structure

```
IMUPoser/
├── README.md
├── install.sh / requirements.txt      # conda env (py3.8, torch 2.x, pytorch_lightning, wandb)
├── live/                              # upstream iOS streaming receiver (unchanged)
├── scripts/
│   ├── train.py                       # Lightning training / fine-tuning entry point
│   ├── evaluate.py                    # full-sequence inference -> predictions/<datasets>/<combo>/<id>.npz
│   ├── run_train.sh, run_finetune_real.sh, run_eval.sh
│   └── 3. Dataset Walkthrough.ipynb   # upstream notebook on the released IMUPoser dataset
└── src/imuposer/
    ├── config.py                      # Config (datasets, window, lr, ...), amass_combos, joint groups
    ├── utils.py                       # get_parser, convert_subset_pose_to_full
    ├── imu4d/                         # runtime access to IMU4D data (identical copy in MobilePoser)
    │   ├── paths.py                   # IMU4D_ROOT / IMU4D_DATA_ROOT / SMPLX_MODEL_PATH resolution
    │   ├── wds_reader.py              # IMU4DShardDataset: streaming IterableDataset over tar shards
    │   └── sample.py                  # convert_sample: smpl85 -> pose/transl, imu_traj -> acc/ori
    ├── datasets/
    │   ├── imu4dDataset.py            # IMU4DDataset -> (imu (T,60), pose_r6d (T,132)) [+ meta]
    │   └── utils.py                   # pad_seq collate, IMUPoserDataModule
    ├── models/
    │   ├── utils.py                   # get_model (fresh or from --pretrained_ckpt)
    │   └── LSTMs/
    │       ├── RNN.py                 # Linear -> 2-layer BiLSTM(512) -> Linear
    │       └── IMUPoser_Model.py      # LightningModule: masked MSE + SMPL-X joint-position loss
    ├── smpl/smplxModel.py             # SMPLXBodyModel: 22-joint FK/IK, LBS mesh with pose blend shapes
    └── math/                          # TransPose rotation / spatial utilities (unchanged)
```

### Runtime data pipeline (`imu4d/`)

1. `wds_reader.IMU4DShardDataset` lists the shards of each spec from `manifest.json` (honouring
   `shared_with` for val/test), assigns shards round-robin to DataLoader workers (or, with fewer
   shards than workers, every worker streams all shards and keeps every *n*-th sample), streams
   tar members with `tarfile` in `r|` mode, and shuffles through a bounded buffer. Shard order is
   reshuffled every epoch. There is no `__len__`; `num_samples` comes from the manifests.
2. `sample.convert_sample` parses `motion_data_smpl85` as
   `[root aa 0:3 | 21 body joints aa 3:66 | zeros | pelvis position 72:75 | betas]`, converts to
   `pose (T,22,3,3)` local rotations, crops a random `window_length` window for training, aligns
   the clip to its first frame (yaw about +Y removed, pelvis at the origin, like IMU4D's loader),
   reorders the six IMU4D sensors `[l_hip, r_hip, l_ear, r_ear, l_elbow, r_elbow]` into the
   baseline slots `[lw, rw, lp, rp, h]` with `[4,5,0,1,2,3]`, and either simulates clean IMU
   readings with IMU4D's `simulate_imu_readings` (virtual sets) or uses the measured `imu_acc` /
   `imu_ori` (real sets, 3-tap smoothing, no gravity added). Output acc is divided by 30.
3. `datasets.IMU4DDataset` applies a random device combo (training) or a fixed one (eval) and
   emits `(imu (T,60), pose_r6d (T,132))`; `return_meta=True` adds `(fname, pose_rotmat, transl)`.

### Model and evaluation

* Input `60 = 5 slots x (3 acc + 9 rotmat)`, output `132 = 22 SMPL-X body joints x 6D`. The loss
  is MSE on 6D rotations plus MSE on root-relative joint positions from `SMPLXBodyModel`, both
  masked to real (unpadded) frames.
* `SMPLXBodyModel` reads `SMPLX_NEUTRAL.npz`; joints are the first 22 SMPL-X joints (same order
  and parents as SMPL 0-21). Joint positions match the `smplx` package to <1e-6 m; the mesh uses
  the 55-joint skinning weights with identity for hands / face and applies pose blend shapes.
* `scripts/evaluate.py` writes `pred_rotations / gt_rotations (T,22,3,3)`, `pred_joints /
  gt_joints (T,22,3)` and `gt_root_translation (T,3)`; `pred_root_translation` is zeros because
  IMUPoser does not predict translation. Use `--max-frames 60` (200 for `imuposer/v2` and
  `dipimu/v2`) in `metric.motion`.

## 5. IMUPoser Dataset
The IMUPoser Dataset is available for download [here](https://www.dropbox.com/s/myq9k0x9i5waphn/imuposer_dataset.zip?dl=0). Check out the [Dataset Walkthrough notebook](scripts/3.%20Dataset%20Walkthrough.ipynb) for a quick overview of the data. Big thanks to Justin Macey from CMU and Giorgio Becherini from the Perceiving Systems group at the Max Planck
Institute for Intelligent Systems for help with processing our motion capture data.

## Follow up Research:
- [WheelPoser: Sparse-IMU Based Body Pose Estimation for Wheelchair Users](https://github.com/axle-lab/WheelPoser) (ASSETS 2024)
- [MobilePoser: Real-Time Full-Body Pose Estimation and 3D Human Translation from IMUs in Mobile Consumer Devices](https://spice-lab.org/projects/MobilePoser/) (UIST 2024)
- [SmartPoser: Arm Pose Estimation with a Smartphone and Smartwatch Using UWB and IMU Data](https://www.figlab.com/research/2023/smartposer) (UIST 2023)

## Disclaimer
```
THE PROGRAM IS DISTRIBUTED IN THE HOPE THAT IT WILL BE USEFUL, BUT WITHOUT ANY WARRANTY. IT IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW THE AUTHOR WILL BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
```

## Acknowledgments
Some of the modules in this repo were inspired by the amazing [TransPose](https://github.com/Xinyu-Yi/TransPose/) github repo. If you liked IMUPoser, you should definitely check out TransPose!

## License
The IMUPoser code can only be used for research i.e., non-commercial purposes. For a commercial license, please contact Vimal Mollyn, Karan Ahuja and Chris Harrison.

## Contact
Feel free to contact [Vimal Mollyn](mailto:ms123vimal@gmail.com) for any help, questions or general feedback!
