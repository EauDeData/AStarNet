
pip install numpy==1.26.4
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
git clone git@github.com:anilakash/torchdrug.git
cd torchdrug
pip install -r requirements.txt
python setup.py install
cd ..; rm -rf ./torchdrug


pip install ogb
pip install easydict
pip install PyYAML
pip install easydict
