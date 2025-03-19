# Multilingual_Lang_Mapping
Multilingual Natural Languages and Programming Languages Mapping

## Instructions for running the code:
`python -m venv ./venv`

`source venv/bin/activate` or `source venv/Scripts/activate`

### Then, install the libraries from requirements.txt:
`pip install requirements.txt`

Don't forget to add your API key in an environmental variable source file and name it `.venv`

### Args taken for `mceval_translator_base.py`
- dataset: HuggingFace dataset to be processed
- subset_name: The name of the split/subset of the dataset processed
- start_idx: The start index of the row of data you wish to start processing in the dataset
- end_idx: The end index of the row of data you wish to stop processing in the dataset
- target_languages: List of target languages for translation
- max_iterations: Maximum number of iterations for translation
- save_interval: The interval you wish to setup for saving intermediate results
- rescaling: Enable rescaling with normalized BERTScore. Rescale on to normalize BERTScore to 0-1, default score is 0.7-1

Currently in the code the sleep time between each translation is set to randomized between 1 to 2 seconds to avoid hitting the rate limit. Modify it base on your own preference.

### An example of the script running command without rescaling:
`python mceval_translator_base.py --dataset generation --start_idx 0 --end_idx 5 --languages Spanish French German --max_iterations 3 --save_interval 2`

### An example of the script running command with rescaling on:
`python mceval_translator_base.py --dataset generation --start_idx 0 --end_idx 5 --languages Spanish French German --max_iterations 3 --save_interval 2 --rescaling`
