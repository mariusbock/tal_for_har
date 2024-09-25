# Installation Guide
Clone repository:

```
git clone anonymized
cd tal_for_har
```

Create [Anaconda](https://www.anaconda.com/products/distribution) environment:

```
conda create -n tal_for_har python==3.10.4
conda activate tal_for_har
```

Install PyTorch distribution:

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install other requirements:
```
pip install -r requirements.txt
sudo apt-get install texlive-latex-extra texlive-fonts-recommended cm-super
```

Compile C++ distributions of NMS (used by ActionFormer, TemporalMaxer and Tridet)

```
cd camera_baseline/actionformer/libs/utils
python setup.py install --user
cd ../../../..
cd camera_baseline/temporalmaxer/libs/utils
python setup.py install --user
cd ../../../..
cd camera_baseline/tridet/libs/utils
python setup.py install --user
cd ../../../..
```

