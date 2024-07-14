conda create -n intermediate-task-selection python=3.8.16
conda activate intermediate-task-selection
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia 
conda install -c conda-forge cxx-compiler
pip install pyarrow
pip install dill pandas tqdm==4.27.0
pip install click
pip install urllib3
pip install charset-normalizer==2
pip install idna==2.5
python setup.py install

