## Bestest little Higgs and Machine Learning framework


Code to simulate Higgs physics beyond the standard model with Pythia8

### Framework installation

To install the framework you need anaconda and git on a linux machine. In a terminal type:
1. Clone the repository:
  ```
  git clone git@github.com:andrex-naranjas/higgsbsm-fw
  ```
2. Access the code:
  ```
  cd higgsbsm-fw
  ```
3. Install the conda enviroment:
  ```
  conda env create -f config.yml
  conda activate hep-ml
  conda develop .
  ```
3.1 Update the conda enviroment:
   ```
   conda env update --file config.yml --prune
   ```
