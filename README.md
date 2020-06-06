# AJ-Summarisation

A tool to generate summaries.

## Installation

1. Use a vritual environment using either virtualenv or anaconda
   - a. virtualenv:
    ```
    sudo pip3 install -U virtualenv
    virtualenv --system-site-packages -p python3 ./venv
    source ./venv/bin/activate  # sh, bash, ksh, or zsh
    ```
   - b. anaconda:
    Download anaconda or miniconda from their website
    ```
    conda create -n tfg  tensorflow-gpu  tensorflow=2
    conda activate tfg
    ```
2. Download the following packages and tools (we used `pip install`):
```
  rouge
  nltk
  tensorflow
  tensorflow_hub
  tqdm
  sklearn
  scipy
  numpy
  pickle
```

## Usage

1. Make sure to be in virtual environment
  - source ./venv/bin/activate  # sh, bash, ksh, or zsh
2. cd to AJ-summarization
3. python3 summarization.py
4. cd evaluation_src
5. python3 evaluation.py

## Contributing

Accepting No Pull Requests For Now.

## License
[MIT](https://choosealicense.com/licenses/mit/)