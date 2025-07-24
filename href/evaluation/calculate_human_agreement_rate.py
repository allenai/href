import os
import json
import argparse
import logging
import random
from collections import defaultdict
import datasets
from alpaca_eval import evaluate as alpaca_farm_evaluate
from href.evaluation.evaluators import DEFINED_ANNOTATORS, ANNOTATOR_SUITE_DICT, ANNOTATION_REVERSE_MAP
import href.evaluation.evaluators as annotator_funcs
from collections import Counter


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
    assert args.annotator is not None and args.config_dir is not None, "Please specify the configuration of the annotator."

    # load model responses
    href_data = datasets.load_dataset(args.dataset)[args.split]
    data = defaultdict(list)
    for example in href_data:
        category = example['category']
        if args.nr_category and category not in args.nr_category:
            continue
        data[category].append(example)

    # specify the annotator for each category
    if args.annotator in ANNOTATOR_SUITE_DICT: # using different annotators for different category
        for category in args.nr_category:
            assert category in ANNOTATOR_SUITE_DICT[args.annotator], \
            f"Category {category} does not have an assigned annotator by {args.annotator}."
        category_to_annotator = ANNOTATOR_SUITE_DICT[args.annotator]
    else: # one single annotator for all categories
        category_to_annotator = {category: {
                                    'annotator': args.annotator, 
                                    'use_human_ref': args.use_human_reference} 
                                for category in args.nr_category}

    # running evaluation through AlpacaEval
    for category in args.nr_category:
        annotator = category_to_annotator[category]['annotator']
        use_human_reference = category_to_annotator[category]['use_human_ref']

        category_responses_a = [{
            "instruction": dp['instruction'],
            "output": dp['output_a'],
            "generator": dp['generator_a'],
            # "perplexity": dp['perplexity_a'],
            "dataset": f"href_{category}"
        } for dp in data[category]]
        category_responses_b = [{
            "instruction": dp['instruction'],
            "output": dp['output_b'],
            "generator": dp['generator_b'],
            # "perplexity": dp['perplexity_b'],
            "dataset": f"href_{category}"
        } for dp in data[category]]
        if use_human_reference:
            category_human_references = [{
                "instruction": dp['instruction'],
                "output": dp['reference'],
                "generator": "human",
                "dataset": f"href_{category}"
            } for dp in data[category]]
        else:
            category_human_references = None
        logging.info(f"Running evaluation on category: {category}")
        output_path = os.path.join(args.save_dir, "human_agreement_analysis", category.lower().replace(" ", "_"))
        os.makedirs(output_path, exist_ok=True)        

        if annotator == "perplexity":
            cur_annotations = annotate_based_on_perplexity(category_responses_b, category_responses_a, category_human_references)
            os.makedirs(os.path.join(output_path, annotator), exist_ok=True)
            json.dump(cur_annotations, open(os.path.join(output_path, annotator, "annotations.json"), 'w')) 
        elif annotator in DEFINED_ANNOTATORS: # non-llm annotators
            # run the according evaluation function
            evaluate_func = getattr(annotator_funcs, annotator)
            cur_annotations = evaluate_func(category_responses_b, category_responses_a, category_human_references, args)
            os.makedirs(os.path.join(output_path, annotator), exist_ok=True)
            json.dump(cur_annotations, open(os.path.join(output_path, annotator, "annotations.json"), 'w'))
        else: # llm annotators
            cache_dir = os.path.join(args.cache_dir, "human_agreement_analysis", category.lower().replace(" ", "_"))
            os.makedirs(cache_dir, exist_ok=True)
            alpaca_farm_evaluate(
                model_outputs=category_responses_a,
                reference_outputs=category_responses_b,
                human_outputs=category_human_references,
                annotators_config=annotator,
                output_path=output_path,
                is_return_instead_of_print=True,
                caching_path=os.path.join(cache_dir, f"{annotator}.json"),
                precomputed_leaderboard=None,
                is_cache_leaderboard=False,
                base_dir=args.config_dir,
                seed=args.seed,
                output_keys=("output_1", "output_2", "output_human") if use_human_reference else ("output_1", "output_2")
            )
            cur_annotations = json.load(open(os.path.join(output_path, annotator, "annotations.json"), 'r'))
        
        # we combined the results if coming from different basic annotators
        if annotator != args.annotator: 
            os.makedirs(os.path.join(output_path, args.annotator), exist_ok=True)
            json.dump(cur_annotations, open(os.path.join(output_path, args.annotator, "annotations.json"), 'w'))


        # calculate agreement with human
        cur_annotations = json.load(open(os.path.join(output_path, args.annotator, "annotations.json"), 'r'))
        prompt_to_annotation = {}
        for a in cur_annotations:
            prompt_to_annotation[a["instruction"], a["generator_2"], a["generator_1"]] = ANNOTATION_REVERSE_MAP[int(a["preference"])] \
                if a["preference"] != None else -1 
        rates_inner = []
        rates_outer = []
        for dp in data[category]:
            rates_inner.append(leave_one_out_agreement_inner(dp['annotations']))
            rates_outer.append(leave_one_out_agreement_outer(dp['annotations'], prompt_to_annotation[dp['instruction'], dp['generator_a'], dp['generator_b']]))
        agreement_rate_inner = sum(rates_inner) / len(rates_inner)
        agreement_rate_outer = sum(rates_outer) / len(rates_outer)
        logging.info(f"Category: {category}")
        logging.info(f"Inner human agreement rate: {agreement_rate_inner * 100 : .1f}%")
        logging.info(f"Human agreement rate using {args.annotator}: {agreement_rate_outer * 100 : .1f}%")
        
    
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
        "--save_dir",
        type=str, 
        default="results",
        help="Directory to save all results"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="The directory to store downloaded datasets, models, and intermmediate annotation files.",
    )
    # evaluation arguments
    parser.add_argument(
        "--annotator",
        type=str,
        default="href",
        help="Name of the evaluation methods. It has to be one the three following: 1. a basic annotator defined in evaluation/evaluators.DEFINED_ANNOTATORS. 2. a configuration name for llm_as_a_judge that corresponds to a directory in llm_as_a_judge. 3. a suite of the above two types of unit evaluators defined in evaluation/evaluators.DEFINED_ANNOTATOR_SUITE_DICT`."
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="href/llm_as_a_judge/configs",
        help="The directory to contain configures for llm_as_a_judge evaluators",
    )
    parser.add_argument(
        "--use_human_reference",
        action="store_true",
        help="Whether of not annotator needs to use the human reference. No need to specify if annotator specifies a evaluator suite."
    )

    args = parser.parse_args()
    
    # set up
    random.seed(args.seed)
    os.environ['HF_HOME'] = args.cache_dir

    evaluate(args)

if __name__ == "__main__":
    main()