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

## Data

To fully run this project you will need to individually obtain the data from [Cornell NEWSROOM](http://lil.nlp.cornell.edu/newsroom/download/index.html).
After downloading and unzipping the data, make the directory ~/AJ-summarization/input_data and place the train, dev, and test files in that folder.

## Usage

1. Make sure to be in virtual environment
2. cd to AJ-summarization
3. `python3 [summarization strategy file]` (listed below)
4. cd evaluation_src
5. python3 evaluation.py

List of summarization strategies:

`summarization-baseline.py` : Takes the first sentence of each article.

`summarization.py` : Uses our modified TextRank to generate the summary.

`first-bias.py` : First bias technique

`weighted.py` : Statistical weighting strategy

In the future, we plan to add a bash file that can run the desired strategies and evaluation in one command.

## Contributing

Accepting No Pull Requests For Now.

## License
[MIT](https://choosealicense.com/licenses/mit/)