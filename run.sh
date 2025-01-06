export HF_TOKEN=$(bash -ic 'source ~/.bashrc; echo $HF_TOKEN')

. ~/venv/bin/activate
pip install datasets
python notebooks/sft_trainer.py