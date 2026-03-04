# # LINGO / global (5-pt)
# python evaluate.py \
#     --dataset lingo \
#     --combo global \
#     --checkpoint '../../checkpoints/IMUPoserGlobalModel_all_global-02272026-112420/epoch=epoch=19-val_loss=validation_step_loss=0.01322.ckpt'

# # LINGO / lw_rp_h (3-pt)
# python evaluate.py \
#     --dataset lingo \
#     --combo lw_rp_h \
#     --checkpoint '../../checkpoints/IMUPoserGlobalModel_all_global-02272026-112420/epoch=epoch=19-val_loss=validation_step_loss=0.01322.ckpt'

# # HumanML / global (5-pt)
# python evaluate.py \
#     --dataset humanml \
#     --combo global \
#     --checkpoint '../../checkpoints/IMUPoserGlobalModel_all_global-02272026-112420/epoch=epoch=19-val_loss=validation_step_loss=0.01322.ckpt'

# # HumanML / lw_rp_h (3-pt)
# python evaluate.py \
#     --dataset humanml \
#     --combo lw_rp_h \
#     --checkpoint '../../checkpoints/IMUPoserGlobalModel_all_global-02272026-112420/epoch=epoch=19-val_loss=validation_step_loss=0.01322.ckpt'


# checkpoints trained on SMPL-X data (LINGO only)
python evaluate.py --dataset smplx --combo global --checkpoint ../../checkpoints/IMUPoserGlobalModel_smplx_global-03032026-195108/epoch=epoch=43-val_loss=validation_step_loss=0.02013.ckpt

python evaluate.py --dataset smplx --combo lw_rp_h --checkpoint ../../checkpoints/IMUPoserGlobalModel_smplx_global-03032026-195108/epoch=epoch=43-val_loss=validation_step_loss=0.02013.ckpt