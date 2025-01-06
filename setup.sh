sudo apt-get install -y -qq software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y -qq python3.11-full python3.11-dev

python3.11 -m venv ~/venv
. ~/venv/bin/activate

python get-pip.py
pip install -U wheel

pip install -e .
# sudo docker build -t easydel-base -f Dockerfile .
