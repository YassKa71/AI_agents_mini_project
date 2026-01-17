"""
Finetuning dataset "research\datasets\environment_datasets\finetuning_dataset.jsonl" construction.
"""

import json
import copy
import yaml

import pandas as pd

from pathlib import Path
from jinja2 import Template
from research.paths import RESEARCH_REPO_ROOT



def generate_system_prompt(task_id, clear_df, prompts):
    sample = clear_df.iloc[int(task_id)]
    task = sample.loc["instruction"]
    tools = json.loads(sample.loc["tools"])
    input_ = sample.loc["input"]
    system_prompt = prompts["system_prompt"]
    messages = []
    for message in system_prompt:
        template = Template(message["content"])
        rendered_content = template.render(input=input_, tools=tools, task=task)
        messages.append({"role": message["role"], "content": rendered_content})
    return messages

def generate_completion(dialog):
    completion = None
    tool_call = {
        "name": None,
        "arguments": []
    }
    if "tool_calls" in dialog.keys():
        thought = "Thought:\n"
        if "thought" in dialog.keys():
            thought += dialog["thought"]
        action = "Action:\n"
        tool_call["name"] = dialog["tool_calls"][0]["function"]["name"]
        for name, value in dialog["tool_calls"][0]["function"]["arguments"].items():
            argument = {"name": name, "value": value}
            tool_call["arguments"].append(argument)
        action += str(tool_call)
        completion = {"role": "assistant", "content": thought + "\n" + action}
    return completion

def generate_observation(observation):
    return {"role": "user", "content": observation}

def prompt_completion_per_task(dialogs, task_id, system_prompt, clear_df):
    """
        Generate a list of prompt completion samples for each task.

        - Sample 1: {
            "prompt": System prompt rendered using task question, tools and inputs
            "completion": First Assistant response in the dialog
        }
        - Sample 2: {
            "prompt": prompt and completion of Sample 1
            "completion": Second Assistant response in the dialog
        }
        etc... until we reach the final answer.

    """
    samples = []
    history = system_prompt
    for dialog in dialogs[1:]:
        if dialog["role"] == "tool":
            observation = generate_observation(dialog["content"]["content"])
            history.append(observation)
        elif dialog["role"] == "assistant":
            completion = generate_completion(dialog)
            if completion:
                sample = {"prompt": copy.deepcopy(history), "completion": [completion], "task_id": task_id}
                samples.append(sample)
                history.append(completion)
    sample_clear_df = clear_df.iloc[int(task_id)]
    if sample_clear_df["final_answer"]:
        completion = "Action:\n"
        final_answer = {
            "name": "FinalAnswer",
            "arguments": [{"name": "answer", "value": sample_clear_df["final_answer"]}]
        }
        completion += str(final_answer)
        completion = {"role": "assistant", "content": completion}
        sample = {"prompt": copy.deepcopy(history), "completion": [completion], "task_id": task_id}
        samples.append(sample)

    return samples


def generate_prompt_completion_dataset(raw_data, clear_df, out_path, prompts):
    " Generate a jsonl file containing all prompt completion samples"
    # creating an output file
    out_path = Path(out_path)
    all_samples = []
    for task_id, task in raw_data.items(): #iterate over tasks
        system_prompt = generate_system_prompt(task_id, clear_df, prompts)
        dialogs = task["dialogs"]
        samples = prompt_completion_per_task(dialogs, task_id, system_prompt, clear_df)
        all_samples += samples
    # write each sample in a separate line in the output file
    with out_path.open("w", encoding="utf-8") as out_f:
        for sample in all_samples:
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    # load raw GTA dataset
    path = RESEARCH_REPO_ROOT / "datasets" / "environment_datasets" / "raw_gta_dataset.json"
    with path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)
    # load clear GTA dataset
    clear_gta_path = RESEARCH_REPO_ROOT / "datasets" / "environment_datasets" / "clear_gta.csv"
    clear_df = pd.read_csv(clear_gta_path)
    # load prompt files
    prompts_path = RESEARCH_REPO_ROOT / "prompts" / "ReAct_prompts_finetuning.yaml"
    with open(prompts_path) as file:
        prompts = yaml.safe_load(file)
    # create finetuning dataset
    out_path = RESEARCH_REPO_ROOT / "datasets" / "environment_datasets" / "finetuning_dataset.jsonl"
    generate_prompt_completion_dataset(raw_data, clear_df, out_path, prompts)

