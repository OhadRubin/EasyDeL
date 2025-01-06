export HF_TOKEN=$(bash -ic 'source ~/.bashrc; echo $HF_TOKEN')

. ~/venv/bin/activate
# pip install datasets
# pip install --find-links https://download.pytorch.org/whl/torch_stable.html torch==2.3.1+cpu
python notebooks/sft_trainer.py