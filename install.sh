conda create -n "imuposer" python=3.8
conda activate imuposer
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
python -m pip install -r requirements.txt
python -m pip install -e src/
pip install --upgrade wandb