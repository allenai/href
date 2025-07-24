import os
import json
import argparse
import logging
import random
from collections import defaultdict
import torch
import datasets
import vllm
from openai import OpenAI
import yaml

from href.generation.utils import generate_completions, create_prompt_with_huggingface_tokenizer_template, load_hf_lm, load_hf_tokenizer


def generate(args):
    assert args.model_name is not None, "Model name should be specified."
    model_name = args.model_name
    config_path = os.path.join(args.generation_config_dir, f"{model_name}.yaml")
    assert os.path.exists(config_path), f'Did not find generation configuration at {config_path}'
    # read in the yaml file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # we skip everything if all outputs have been generate
    need_to_load_model = False
    for category in args.nr_category:
        save_path = os.path.join(args.save_dir, model_name, category.lower().replace(" ", "_"), "responses.jsonl")
        need_to_load_model = need_to_load_model or not os.path.exists(save_path)
    if not need_to_load_model:
        logging.info("Found all saved generations, reusing the generations.")
        return


    # load href from huggingface
    raw_text_prompts = defaultdict(list)  # category -> list of example dicts
    href_data = datasets.load_dataset(args.dataset)[args.split]
    for example in href_data:
        category = example['category']
        if args.nr_category and category not in args.nr_category:
            continue
        raw_text_prompts[category].append(example['instruction'])
    

    # prepare the input and config
    if "openai" not in config: # local model
        # we always load the tokenizer for vllm or hf models
        tokenizer = load_hf_tokenizer(
            model_name_or_path=config['model_name_or_path'],
            tokenizer_name_or_path=config['tokenizer_name_or_path'],
            use_fast_tokenizer=not config['use_slow_tokenizer'],
        )
        if config['use_vllm']: # load vllm
            vllm_model = vllm.LLM(
                model=config['model_name_or_path'],
                tokenizer=config['tokenizer_name_or_path'] if config['tokenizer_name_or_path'] is not None else config['model_name_or_path'],
                tensor_parallel_size=torch.cuda.device_count(),
                download_dir=f"{args.cache_dir}/models"
            )
            sampling_params = vllm.SamplingParams(
                temperature=config['temperature'],  # greedy decoding
                max_tokens=config['max_new_tokens']
            )
        else: # load hf model
            model = load_hf_lm(
                model_name_or_path=config['model_name_or_path'],
                load_in_8bit=config['load_in_8bit'],
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=config['gptq'],
            )
            # modify tokenizer if required
            from transformers import GPTNeoXForCausalLM, OPTForCausalLM
            if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
                tokenizer.model_max_length = model.config.max_position_embeddings
                logging.info("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))

        # apply chat format
        if 'format' in config:
            prompts = {}
            sample_prompt = create_prompt_with_huggingface_tokenizer_template([{"role": "user", "content": "<instruction>"}], tokenizer, add_bos=False)
            logging.info(f"Applying chat formatting\n: {sample_prompt}")
            for category, category_prompts in raw_text_prompts.items():
                formatted_prompts = []
                for prompt in category_prompts:
                    if config['format'] == 'default':
                        messages = [{"role": "user", "content": prompt}]
                        prompt = create_prompt_with_huggingface_tokenizer_template(messages, tokenizer, add_bos=False)
                        formatted_prompts.append(prompt)
                    else:
                        formatted_prompts.append(config['format'].format(prompt))
                prompts[category] = formatted_prompts
        else:
            prompts = dict(raw_text_prompts)
    else: # load openai model
        openai_client = OpenAI()
        prompts = dict(raw_text_prompts)


    logging.info("Stats:")
    for category, category_prompts in prompts.items():
        logging.info(f"{category}\t{len(category_prompts)}")

    # generate outputs
    for category, category_prompts in prompts.items():
        assert category in args.nr_category

        # config saving path
        save_dir = os.path.join(args.save_dir, model_name, category.lower().replace(" ", "_"))
        save_path = os.path.join(save_dir, "responses.jsonl")
        if os.path.exists(save_path):
            logging.info(f"Found saved generations for {category}, reusing the generations.")
            continue
        else:
            os.makedirs(save_dir, exist_ok=True)

        logging.info(f"Running inference on category: {category}")
        if not config.get('openai', False): # local model
            if config['use_vllm']:
                logging.info(f"Using VLLM:{sampling_params}")
                category_outputs = vllm_model.generate(category_prompts, sampling_params)
                category_outputs = [it.outputs[0].text for it in category_outputs]
            else:
                generation_kwargs = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'prompts': category_prompts,
                    'max_new_tokens': config['max_new_tokens'],
                    'do_sample': config['temperature'] != 0.0,
                    'batch_size': config['batch_size'] if config['batch_size'] else 1,
                }
                if config['temperature'] != 0.0:
                    generation_kwargs['temperature'] = config['temperature']
                category_outputs = generate_completions(
                    **generation_kwargs
                )
        else: # openai model generation
            assert 'format' not in config
            category_outputs = []
            for prompt in category_prompts:
                response = openai_client.chat.completions.create(
                    model=config['model_name_or_path'],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config['max_new_tokens'],
                    temperature=config['temperature'],
                )
                category_outputs.append(response.choices[0].message.content)
                
        # save the outputs
        with open(save_path, "w") as fout:
            for prompt, output in zip(raw_text_prompts[category], category_outputs):
                example = {
                    "instruction": prompt,
                    "output": output,
                    "generator": f"{model_name}",
                    "dataset": f"href_{category}"
                }
                fout.write(json.dumps(example) + "\n")


def main():
    parser = argparse.ArgumentParser()
    # general arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="The model name that corresponds to the name of the yaml configuration file under `generation_config_dir` (exclude `.yaml`).",
    )
    parser.add_argument(
        "--generation_config_dir",
        type=str,
        default="href/generation/configs",
        help="The directory that contains the model generation configuration files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="allenai/href",
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

    args = parser.parse_args()

    # set up
    random.seed(args.seed)
    os.environ['HF_HOME'] = args.cache_dir

    generate(args)

if __name__ == "__main__":
    main()