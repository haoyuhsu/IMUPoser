# Fine-tune on DIP-IMU only
python "2. FineTune RealIMU.py" \
    --combo_id global \
    --experiment IMUPoserFineTuneRealIMU_dip \
    --pretrained_ckpt /home/haoyuyh3/Documents/maxhsu/imu-humans/IMUPoser/checkpoints/IMUPoserGlobalModel_all_global-02272026-112420/epoch=epoch=19-val_loss=validation_step_loss=0.01322.ckpt \
    --dataset_name dip \
    --max_epochs 30

# Fine-tune on IMUPoser real-world only
python "2. FineTune RealIMU.py" \
    --combo_id global \
    --experiment IMUPoserFineTuneRealIMU_imuposer \
    --pretrained_ckpt /home/haoyuyh3/Documents/maxhsu/imu-humans/IMUPoser/checkpoints/IMUPoserGlobalModel_all_global-02272026-112420/epoch=epoch=19-val_loss=validation_step_loss=0.01322.ckpt \
    --dataset_name imuposer_real \
    --max_epochs 50