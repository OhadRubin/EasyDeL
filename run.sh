# export HF_TOKEN=$(bash -ic 'source ~/.bashrc; echo $HF_TOKEN')
# . ~/venv/bin/activate

# python3.10 -m pip install -U "jax[tpu]" "flax[all]" numpy~=1.0 eformer==0.0.4 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 
# python3.10 -m pip install --upgrade   uvloop==0.21.0 uvicorn==0.32.0 jinja2>=3.1.5 transformers>=4.47.0 prometheus_client>=0.21.0 fastapi>=0.115.2

python3.10 -m pip install -U "jax[tpu]~=0.4" "flax[all]" numpy~=1.0 eformer==0.0.4 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 

python3.10 -m pip install --force-reinstall protobuf~=3.20 "jax[tpu]==0.4.30" "flax[all]" numpy~=1.0  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 




python3.10 -m pip install --force-reinstall wandb
python3.10 -m pip install protobuf~=3.20