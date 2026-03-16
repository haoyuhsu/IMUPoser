# Evaluate fine-tuned model on DIP test set
python evaluate_real_imu.py \
    --dataset dip \
    --pretrained_ckpt /home/haoyuyh3/Documents/maxhsu/imu-humans/IMUPoser/checkpoints/IMUPoserGlobalModel_all_global-02272026-112420/epoch=epoch=19-val_loss=validation_step_loss=0.01322.ckpt \
    --finetuned_ckpt /home/haoyuyh3/Documents/maxhsu/imu-humans/IMUPoser/checkpoints/IMUPoserFineTuneRealIMU_dip_global-03042026-110855/epoch=epoch=24-val_loss=validation_step_loss=0.01590.ckpt \
    --combo lw_rp_h


# Evaluate on IMUPoser test set
python evaluate_real_imu.py \
    --dataset imuposer_real \
    --pretrained_ckpt /home/haoyuyh3/Documents/maxhsu/imu-humans/IMUPoser/checkpoints/IMUPoserGlobalModel_all_global-02272026-112420/epoch=epoch=19-val_loss=validation_step_loss=0.01322.ckpt \
    --finetuned_ckpt /home/haoyuyh3/Documents/maxhsu/imu-humans/IMUPoser/checkpoints/IMUPoserFineTuneRealIMU_imuposer_global-03042026-111021/epoch=epoch=38-val_loss=validation_step_loss=0.01736.ckpt \
    --combo lw_rp_h