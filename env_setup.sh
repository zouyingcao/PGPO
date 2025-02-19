# 1. Install Python dependencies
conda create -n PGPO python==3.9
conda activate PGPO

cd textworld
chmod +x setup.sh
pip install .
cd ..

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r PGPO_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install "fschat[model_worker]"
pip install -e ".[train]"

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.2/flash_attn-2.6.2+cu118torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

cd envs/webshop
pip install -e .
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.6.0/en_core_web_lg-3.6.0-py3-none-any.whl
conda install -y -c conda-forge openjdk=11

# 2. Download data for WebShop environment
gdown https://drive.google.com/uc?id=1G_0ccLWn5kZE5rpeyAdh_YuoNzvBUjT9
gdown https://drive.google.com/uc?id=11zOUDkJSgGhYin9NxQtG8PVpDsika86y
unzip data.zip
mkdir search_index
unzip indexes.zip -d search_index/

# 3. Download data for ALFWorld environment
cd ../..
cd eval_agent/data/alfworld
gdown https://drive.google.com/uc?id=1y7Vqeo0_xm9d3I07vZaP6qbPFtyuJ6kI
unzip alfworld_data.zip

cd ../../..
