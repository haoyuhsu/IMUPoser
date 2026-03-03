# LINGO / global (5-pt)
python evaluate.py \
    --dataset lingo \
    --combo global \
    --checkpoint '../../checkpoints/IMUPoserGlobalModel_all_global-02272026-112420/epoch=epoch=19-val_loss=validation_step_loss=0.01322.ckpt'

# LINGO / lw_rp_h (3-pt)
python evaluate.py \
    --dataset lingo \
    --combo lw_rp_h \
    --checkpoint '../../checkpoints/IMUPoserGlobalModel_all_global-02272026-112420/epoch=epoch=19-val_loss=validation_step_loss=0.01322.ckpt'

# HumanML / global (5-pt)
python evaluate.py \
    --dataset humanml \
    --combo global \
    --checkpoint '../../checkpoints/IMUPoserGlobalModel_all_global-02272026-112420/epoch=epoch=19-val_loss=validation_step_loss=0.01322.ckpt'

# HumanML / lw_rp_h (3-pt)
python evaluate.py \
    --dataset humanml \
    --combo lw_rp_h \
    --checkpoint '../../checkpoints/IMUPoserGlobalModel_all_global-02272026-112420/epoch=epoch=19-val_loss=validation_step_loss=0.01322.ckpt'