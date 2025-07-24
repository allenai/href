import os
import json
import argparse
import numpy as np
import random
from collections import defaultdict
import datasets
from href.evaluation.evaluators import ANNOTATOR_SUITE_DICT
from collections import Counter
import pandas as pd

ANNOTATION_REVERSE_MAP = {
    1: 2,
    2: 1,
    0: 0
}


def bootstrap(data, num_resamples=5000, statistic=np.mean, seed=None):
    """
    Perform bootstrap resampling on a list of scores.

    Parameters:
    - data (list or array): The data to bootstrap (e.g., list of scores).
    - num_resamples (int): The number of bootstrap samples to generate.
    - statistic (function): The statistic to compute on each resample (e.g., np.mean, np.median).
    - seed (int or None): Seed for the random number generator (optional).

    Returns:
    - bootstrapped_stats (array): An array of the computed statistic for each bootstrap sample.
    """
    if seed is not None:
        np.random.seed(seed)
    
    data = np.array(data)
    bootstrapped_stats = []

    for _ in range(num_resamples):
        # Resample with replacement
        sample = np.random.choice(data, size=len(data), replace=True)
        # Compute the statistic on the resample
        bootstrapped_stat = statistic(sample)
        bootstrapped_stats.append(bootstrapped_stat)

    lower_bound = np.percentile(bootstrapped_stats, 2.5)
    upper_bound = np.percentile(bootstrapped_stats, 97.5)
    return np.array(bootstrapped_stats), lower_bound, upper_bound


def calculate_mode(annotations):
    """Calculate the mode of the annotations. If multiple modes exist, choose one at random."""
    count = Counter(annotations)
    max_freq = max(count.values())
    modes = [key for key, freq in count.items() if freq == max_freq]
    
    # Randomly choose one if there are multiple modes
    return random.choice(modes)

def leave_one_out_agreement_inner(annotations):
    """Compute the leave-one-out agreement for a list of annotations."""
    num_annotations = len(annotations)
    correct_predictions = 0
    
    for i in range(num_annotations):
        # Leave one out
        remaining_annotations = annotations[:i] + annotations[i+1:]
        
        # Calculate the mode of the remaining annotations
        mode = calculate_mode(remaining_annotations)
        
        # Check if the prediction matches the mode
        if mode == annotations[i]:
            correct_predictions += 1
    
    # Return the accuracy (agreement rate) for this set of annotations
    return correct_predictions / num_annotations


def leave_one_out_agreement_outer(annotations, my_p):
    """Compute the leave-one-out agreement for a list of annotations."""
    num_annotations = len(annotations)
    correct_predictions = 0
    
    for i in range(num_annotations):
        # Leave one out
        remaining_annotations = annotations[:i] + annotations[i+1:]
        
        # Calculate the mode of the remaining annotations
        mode = calculate_mode(remaining_annotations)
        
        # Check if the prediction matches the mode
        if mode == my_p:
            correct_predictions += 1
    
    # Return the accuracy (agreement rate) for this set of annotations
    return correct_predictions / num_annotations


def annotate_based_on_perplexity(responses_a, responses_b, human_references):
    annotations = []
    for res1, res2 in zip(responses_a, responses_b):
        if res1["perplexity"] < res2["perplexity"]:
            score = 1.0
        if res1["perplexity"] > res2["perplexity"]:
            score = 2.0
        else:
            score = 0.0
        annotations.append({
            "instruction": res1["instruction"],
            "output_1": res1["output"],
            "generator_1": res1["generator"],
            "output_2": res2["output"],
            "generator_2": res2["generator"],
            "annotator": "perplexity",
            "preference": score
        })
    return annotations


def evaluate(args):

    # load model responses
    href_data = datasets.load_dataset(args.dataset)[args.split]
    data = defaultdict(list)
    for example in href_data:
        category = example['category']
        if args.nr_category and category not in args.nr_category:
            continue
        data[category].append(example)

    annotators = args.annotators
    results = {"annotator": [], "average": [], "confidence_interval": []}
    results.update({c : [] for c in args.nr_category})
    # getting results from each annotator
    for annotator in annotators:
        results["annotator"].append(annotator)
        # specify the annotator for each category
        if annotator in ANNOTATOR_SUITE_DICT: # using different annotators for different category
            for category in args.nr_category:
                assert category in ANNOTATOR_SUITE_DICT[annotator], \
                f"Category {category} does not have an assigned annotator by {annotator}."
            category_to_annotator = ANNOTATOR_SUITE_DICT[annotator]
        else: # one single annotator for all categories
            category_to_annotator = {category: {'annotator': annotator} for category in args.nr_category}

        rates_total = []
        for category in args.nr_category:
            print(f"Processing category {category} with annotator {category_to_annotator[category]['annotator']}")
            annotator = category_to_annotator[category]['annotator']

            output_path = os.path.join(args.result_dir, category.lower().replace(" ", "_"))
            cur_annotations = json.load(open(os.path.join(output_path, annotator, "annotations.json"), 'r'))

            prompt_to_annotation = {}
            for a in cur_annotations:
                prompt_to_annotation[a["instruction"], a["generator_2"], a["generator_1"]] = ANNOTATION_REVERSE_MAP[int(a["preference"])] \
                    if a["preference"] != None else -1
            rates_outer = []
            for dp in data[category]:
                rates_outer.append(leave_one_out_agreement_outer(dp['annotations'], prompt_to_annotation[dp['instruction'], dp['generator_a'], dp['generator_b']]))
            
            if len(rates_outer) <= 0:
                assert False, f"Category: {category}"
            results[category].append(sum(rates_outer) / len(rates_outer))
            rates_total.extend(rates_outer)
        
        results['average'].append(sum(rates_total) / len(rates_total))
        _, upper, lower = bootstrap(rates_total)
        results['confidence_interval'].append(f"{lower:.2f}<{sum(rates_total) / len(rates_total):.2f} <{upper:.2f}")


    # Calculate human statistics
    results["annotator"].append("human")
    rates_total = []
    for category in args.nr_category:
        rates_inner = []
        for dp in data[category]:
            rates_inner.append(leave_one_out_agreement_inner(dp['annotations']))
        results[category].append(sum(rates_inner) / len(rates_inner))
        rates_total.extend(rates_inner)

    results['average'].append(sum(rates_total) / len(rates_total))
    _, upper, lower = bootstrap(rates_total)
    results['confidence_interval'].append(f"{lower:.2f}<{sum(rates_total) / len(rates_total):.2f} <{upper:.2f}")
    
    results = pd.DataFrame(results)
    results.to_csv("human_agreement_analysis_results.csv", sep=",")


def main():
    parser = argparse.ArgumentParser()
    # general arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="allenai/href_preference",
        help="The huggingface dataset name or the path to a local file to use for evaluation."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="The split to use in dataset."
    )
    parser.add_argument(
        "--nr_category",
        type=str,
        default=["Generation", "Open QA", "Brainstorm", "Rewrite", "Summarize",
                 "Classify", "Closed QA", "Extract"],
        nargs="+",
        help="Categories in the HREF to include."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--result_dir",
        type=str, 
        default="results",
        help="Directory to save all results"
    )
    parser.add_argument(
        "--save_dir",
        type=str, 
        default="results_old",
        help="Directory to save all results"
    )
    # evaluation arguments
    parser.add_argument(
        "--annotators",
        type=str,
        default="href",
        nargs="+",
        help="Names of the evaluation methods. Each has to be one the three following: 1. a basic annotator defined in evaluation/evaluators.DEFINED_ANNOTATORS. 2. a configuration name for llm_as_a_judge that corresponds to a directory in llm_as_a_judge. 3. a suite of the above two types of unit evaluators defined in evaluation/evaluators.DEFINED_ANNOTATOR_SUITE_DICT`."
    )

    args = parser.parse_args()
    
    # set up
    random.seed(args.seed)

    evaluate(args)

if __name__ == "__main__":
    main()