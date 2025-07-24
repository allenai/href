import os
import json
import argparse
import random
import numpy as np
from collections import defaultdict
from scipy import stats
from itertools import combinations
import pandas as pd
import yaml
from huggingface_hub import HfApi
from tqdm import tqdm

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


# compute pvalue of the t-test between 
def get_pvalue_from_paired_t_test(annotation1, annotation2):
    ttest_result = stats.ttest_rel(annotation1, annotation2)
    return ttest_result.pvalue


# test if significant
# def decide_is_distinguishable(annotation1, annotation2):
#     return get_pvalue_from_paired_t_test(annotation1, annotation2) <= 0.05

def decide_is_distinguishable(ci1, ci2):
    return ci1[3] > ci2[2] or ci1[2] < ci2[3]


def calculate_win_rate(annotations):
    win_annotations = [cur_a in [2.0] for cur_a in annotations]
    tie_annotations = [cur_a in [0.0] for cur_a in annotations]
    return sum(win_annotations) / len(win_annotations) + sum(tie_annotations) / len(tie_annotations) / 2


def upload(args):
    random.seed(42)
    nr_category = args.nr_category
    result_dir = args.result_dir
    models = args.models
    annotator = args.annotator
    save_dir = args.save_dir
    subset_name = args.subset_name
    api = HfApi()

    # initialize csv file
    os.makedirs(os.path.join(save_dir, subset_name), exist_ok=True)
    results_titles = [c.lower().replace(" ", "_") for c in nr_category] + ["average"] 
    rank_titles = [f"{cat.lower().replace(' ', '_')}_rank" for cat in nr_category] + ["average_rank"]
    confi_titles = [f"{cat.lower().replace(' ', '_')}_confi" for cat in nr_category] + ["average_confi"]

    # read annotations
    positive_rates = defaultdict(list)
    annotations = defaultdict(list)
    confidence_interval=defaultdict(list)
    for model in tqdm(models):
        for category in nr_category:
            category_name = category.lower().replace(' ', '_')
            cur_annotations = json.load(open(os.path.join(result_dir, model, category_name, annotator, "annotations.json"), 'r'))
            cur_annotations = [cur_a['preference'] for cur_a in cur_annotations]
            positive_rate = calculate_win_rate(cur_annotations)
            positive_rates[model].append(positive_rate)
            _, lower, upper = bootstrap(cur_annotations, statistic=calculate_win_rate)
            confidence_interval[model].append((upper-positive_rate, lower-positive_rate, upper, lower))
            cur_annotations = [cur_a if cur_a is not None else -1 for cur_a in cur_annotations]
            annotations[model].append(cur_annotations)

        model_annotations = [a for cat_annotations in annotations[model] for a in cat_annotations]
        average_positive_rate = calculate_win_rate(model_annotations)
        positive_rates[model].append(average_positive_rate)
        _, lower, upper = bootstrap(model_annotations, statistic=calculate_win_rate)
        confidence_interval[model].append((upper-average_positive_rate, lower-average_positive_rate, upper, lower))
        annotations[model].append(model_annotations)

    # calculate ranking
    rankings = defaultdict(list)
    for cat_index, cat in enumerate((nr_category + ["Average"])):
        # sort data by average
        sorted_positive_rates = dict(sorted(positive_rates.items(), key=lambda x: x[1][cat_index], reverse=True))
        sorted_ci = [[m] + confidence_interval[m] for m in sorted_positive_rates]

        # decide distinguishibility ranking
        i = 0
        while i < len(sorted_ci):
            model_name = sorted_ci[i][0]
            cur_confidence_interval = sorted_ci[i][cat_index+1]
            # decide if the following model is distinguishable from the current
            rankings[model_name].append(i+1)
            newly_added = 0
            for j in range(i + 1, len(sorted_ci), 1):
                next_model_name = sorted_ci[j][0]
                next_confidence_interval = sorted_ci[j][cat_index+1]
                if not decide_is_distinguishable(cur_confidence_interval, next_confidence_interval):
                    rankings[next_model_name].append(i+1)
                    newly_added += 1
                else:
                    break
            i += 1 + newly_added

    # add rankings and sort by average finally
    sorted_positive_rates = dict(sorted(positive_rates.items(), key=lambda x: x[1][-1], reverse=True))
    sorted_results = [[m] + rates for m, rates in sorted_positive_rates.items()]
    sorted_rank = [rankings[m] for m in sorted_positive_rates]
    sorted_config = [confidence_interval[m] for m in sorted_positive_rates]

    for result, rank, confi in zip(sorted_results, sorted_rank, sorted_config):
        model = result[0]
        # get model path
        config_path = os.path.join(args.config_dir, f"{model}.yaml")
        assert os.path.exists(config_path), f'Did not find generation configuration at {config_path}.'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        output_json = {"path": config['model_name_or_path']}
        output_json.update({results_titles[i]: round(result[i+1], 4 if "average" in results_titles[i] else 3) for i in range(len(results_titles))})
        output_json.update({title: rank[i] for i, title in enumerate(rank_titles)})
        output_json.update({title: f"+{100*confi[i][0]:.2f} / -{abs(100*confi[i][1]):.2f}" if "average" in title else
            f"+{100*confi[i][0]:.1f} / -{abs(100*confi[i][1]):.1f}"  
            for i, title in enumerate(confi_titles)})
        json.dump(output_json, open(os.path.join(save_dir, subset_name, f"{model}.json"), 'w'), indent=4)

    files = api.list_repo_files(repo_id=args.dataset, repo_type="dataset")
    # Loop through and delete each file
    for file_path in files:
        api.delete_file(path_in_repo=file_path, repo_id=args.dataset, repo_type="dataset")
        print(f"Deleted {file_path}")

    api.upload_folder(
        folder_path=os.path.join(save_dir),
        repo_id=args.dataset,
        repo_type="dataset",
    )
            

def main():
    parser = argparse.ArgumentParser()
    # general arguments
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        nargs="+",
        help="Name of the models to build the leaderboard."
    )
    parser.add_argument(
        "--annotator",
        type=str,
        default='href',
        help="Name of the evaluator that generated the annotation results."
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        default='temperature=0.0',
        help="Name of the evaluator that generated the annotation results."
    )
    parser.add_argument(
        "--nr_category",
        type=str,
        nargs="+",
        default=["Generation", "Open QA", "Brainstorm", "Rewrite", "Summarize",
                 "Classify", "Closed QA", "Extract", 'Reasoning Over Numerical Data',
                    'Multi-Document Synthesis', 'Fact Checking or Attributed QA'],
        help="Categories in the HREF to include."
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Path to the dir that contains the annotation files."
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="href/generation/configs",
        help="Path to the dir that contains the annotation files."
    )
    parser.add_argument(
        "--save_dir",
        type=str, 
        default="results_for_upload",
        help="Directory to save all results."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="alrope/href_results",
        help="Path to the reference outputs. If none is provided, will use human-written references."
    )
    args = parser.parse_args()

    upload(args)

if __name__ == "__main__":
    main()