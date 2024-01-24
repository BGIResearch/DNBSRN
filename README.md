# DNBSRN
A deep learning model for improving the image resolution of ultra-high-density arrays in DNBSEQ.
## Installation (On Windows)
### 1.Requirements
* Python 3.11.3

* Python venv

* Open a command window
### 2.Create a virtual environment

```
#create
python -m venv DNBSRNenv
#activate
cd DNBSRNenv/Scripts && activate
#deactivate when not in use (optional)
deactivate
```

### 3. clone DNBSRN

```
#Download DNBSRN from github
git clone https://github.com/BGIResearch/DNBSRN.git
#Go to the code folder
cd DNBSRN
```

### 4.Install required packages

```
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### 5.Download datasets
* Download training data from https://ftp.cngb.org/pub/CNSA/data2/CNP0005204/CNS0969113/train_image.zip, unzip and place them in 'DNBSRN/train_image'.
* Download test data from https://ftp.cngb.org/pub/CNSA/data2/CNP0005204/CNS0969113/test_image.zip, unzip and place them in 'DNBSRN/test_image'.
## Usage
To train, test, and analyze the efficiency metrics of different networks, execute 'scripts/run.py'.

options:
* -B   Execution of network training
* -C   Execution of network testing
* -E   Analyze the efficiency metrics
* -m   Select which network to train, test, and analyze, choose from {DNBSRN, IMDN, RFDN, RLFN, EDSR, RDN, RCAN, DNBSRN_kernel_size_3, DNBSRN_kernel_size_5, DNBSRN_kernel_size_9, DNBSRN_delete_IIC, DNBSRN_delete_SRB}
* -t   Select which dataset to test, choose from {dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8}
* -g   Choose which gpu to use, default=0
* -p   Whether to perform HM preprocessing when testing, choose from {true, false}, default=true
* -s   Whether to save intermediate results of HM preprocessing, choose from {true, false}, default=false

More parameters can be modified in 'scripts/parameter.py'.


Specifically, to train, test, and analyze the efficiency metrics of DNBSRN

```
#train
python scripts/run.py -B -m DNBSRN
#test
python scripts/run.py -C -m DNBSRN -t dataset1 dataset2 dataset3 dataset4 dataset5 dataset6 dataset7 dataset8
#analyze the efficiency metrics
python scripts/run.py -E -m DNBSRN
```

The 'result' folder contains our trained models.
