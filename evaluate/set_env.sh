pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers --upgrade

# for qwen infer
pip install qwen-vl-utils[decord]==0.0.8

# for internvl infer
pip install timm
pip uninstall opencv-python -y
pip install opencv-python-headless --force-reinstall

# for keye-vl infer
pip install keye-vl-utils==1.5.2

# for seed infer
pip install 'volcengine-python-sdk[ark]'