
import os

# split = 'test'

# humanml_data_root = '/home/haoyuyh3/Downloads/humanml_smpl_files'
# data_dir = os.path.join(humanml_data_root, split)

# file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])

# with open(f'_tmp_fname/humanml_{split}_fname.txt', 'w') as f:
#     for fname in file_list:
#         f.write(f"{fname}\n")


split = 'test'

lingo_data_root = '/home/haoyuyh3/Downloads/lingo_smpl_files'
data_dir = os.path.join(lingo_data_root, split)

file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])

with open(f'_tmp_fname/lingo_{split}_fname.txt', 'w') as f:
    for fname in file_list:
        f.write(f"{fname}\n")