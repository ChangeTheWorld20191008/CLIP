# create env
    mkvirtualenv clip
    conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
    pip3 install ftfy regex tqdm
    pip3 install git+https://github.com/openai/CLIP.git
    pip3 install scikit-image
    sudo apt-get install python3-tk