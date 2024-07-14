import re
import os
import json

def file_exist(fname):
    return os.path.exists(fname)

def get_score(fname, best_metric_key="best_metric"):
    """
    Read json file and return best metric score
    """
    with open(fname, "r") as f:
        data = json.load(f)

    # Get best score
    best_score = None
    try:
        best_score = data[best_metric_key]
    except:
        print(f"Key {best_metric_key} not in file.")
    return best_score

def get_score_using_regex(fname):

    with open(fname, "r") as f:
        data = f.read()

    # Define the regex pattern
    pattern = r"test_accuracy\s*=\s*([\d.]+)"

    # Use regex to extract the test accuracy value
    match = re.search(pattern, data)

    if match:
        test_accuracy = match.group(1)
        print("Test accuracy: ", test_accuracy)
    else:
        print("Test accuracy not found in the input string.")


def main():

    tasks = ["cb"]
    lrs = ["5", "4", "3", "2", "1", "5e-1", "3e-1", "1e-1", "5e-2", "3e-2", "1e-2", "5e-3"]
    lrs = ["3e-1", "1e-1", "5e-2", "3e-2", "1e-2", "5e-3", "3e-3", "1e-3", "3e-4"]
    fname_formatter = "/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-#0/t5-base-lm-adapt/#1/42/trainer_state.json"

    for t in tasks:
        for lr in lrs:
            fname = fname_formatter.replace("#0", lr).replace("#1", t)            
            if file_exist(fname):
                print(f"Reading path: {fname}")
                
                score = get_score(fname)
                #score = get_score_using_regex(fname)
                print(f"best_socre: {score}") 
            else:
                print(f"File does not exist: {fname}")

if __name__ == "__main__":
    main()
