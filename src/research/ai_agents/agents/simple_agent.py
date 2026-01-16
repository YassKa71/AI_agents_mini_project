"""
All simple paradigms of AI agents with a queue-structured memory system.
Structure inspired from toolagent paradigm of HF smolagents:
https://github.com/huggingface/smolagents/tree/9f43bbd8e7b52135f27e8071ce3b6d517d4545fd
"""

import json
import logging
import re
import time

import pandas as pd
import yaml
from jinja2 import Template
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

from research.utils.utils import ActionBlockStoppingCriteria, parse
from research.ai_agents.environment.evaluation import StaticEnvironment
from research.paths import RESEARCH_REPO_ROOT

# Configure logging
logging.basicConfig(level=logging.INFO)


class BaselineAgent:
    """
    This class represents the baseline paradigms to be tested in our experiment.
    The group of paradigms represented by this class are characterized by:
    - A queue memory system where information is appended over time
    - No planning process
    - Support for different prompt files which must include the following prompts: system_prompt,
        observation_prompt, parsing_error_prompt
    """

    def __init__(self, model_name, environment, prompts_file, max_steps=50):
        self.environment = environment
        self.memory = []
        self.max_steps = max_steps
        self.results = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        with open(prompts_file) as file:
            self.prompts = yaml.safe_load(file)
            if not all(key in self.prompts for key in ["system_prompt", "observation_prompt", "parsing_error_prompt"]):
                raise KeyError
        self.OBSERVATION_TYPE_SUCCESS = "success"
        self.stopping_criteria = ActionBlockStoppingCriteria(self.tokenizer)

    def run(self):
        "Main agent loop"
        # initialization
        self.memory = self.generate_system_prompt()

        # start loop
        current_step = 0
        solved = False
        while current_step < self.max_steps and not solved:
            start_time = time.time()
            step_results = {
                "num_step": current_step,
            }
            logging.info(f"Step {current_step}: ")
            logging.info("Running the llm...")
            output = self.run_slm(self.memory, step_results=step_results)
            logging.info(f"llm output: {output}")
            step_results["llm_output"] = output
            logging.info("Parsing the output...")
            try:
                parsed_output = parse(output)
                _, _ = parsed_output["name"], parsed_output["arguments"]
                for argument in parsed_output["arguments"]:
                    _, _ = argument["name"], argument["value"]
                observation = self.run_environment(parsed_output, start_time, step_results)

                if observation == self.OBSERVATION_TYPE_SUCCESS:  # if task fulfilled stop the loop
                    solved = True
                    logging.info("Task Completed !!!")
                    self.results.append(step_results)
                    continue

                tool_observation_prompt = self.generate_observation_prompt(parsed_output, observation)
                self.memory += tool_observation_prompt
                current_step += 1
                self.results.append(step_results)
            except (json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError):
                # use prompt for reformatting
                new_parsed_output = self.reformat(output)
                if new_parsed_output:
                    observation = self.run_environment(new_parsed_output, start_time, step_results)
                    if observation == self.OBSERVATION_TYPE_SUCCESS:  # if task fulfilled stop the loop
                        solved = True
                        logging.info("Task Completed !!!")
                        self.results.append(step_results)
                        continue

                    tool_observation_prompt = self.generate_observation_prompt(new_parsed_output, observation)
                    self.memory += tool_observation_prompt
                    current_step += 1
                    self.results.append(step_results)
                else:
                    # adding duration and num_tokens to step_results
                    duration = time.time() - start_time
                    step_results["duration"] = duration
                    # using parsing_error_prompt as observation
                    parsing_error_prompt = self.prompts["parsing_error_prompt"]
                    for message in parsing_error_prompt:
                        template = Template(message["content"])
                        rendered_content = template.render(output=output)
                        self.memory.append({"role": message["role"], "content": rendered_content})
                    current_step += 1
                    logging.warning("Error while parsing the output.")
                    step_results["observation"] = "parsing error"
                    self.results.append(step_results)
                    continue

    def reformat(self, output):
        messages = []
        reformatting_prompt = self.prompts["reformatting_prompt"]
        for message in reformatting_prompt:
            template = Template(message["content"])
            rendered_content = template.render(output=output)
            messages.append({"role": message["role"], "content": rendered_content})
        new_output = self.run_slm(messages)
        try:
            parsed_output = parse(new_output)
            _, _ = parsed_output["name"], parsed_output["arguments"]
            for argument in parsed_output["arguments"]:
                _, _ = argument["name"], argument["value"]
        except (json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError):
            return None
        return parsed_output

    def run_environment(self, parsed_output, start_time, step_results):
        # adding duration to step_results
        duration = time.time() - start_time
        step_results["duration"] = duration

        logging.info("Output parsed successfully! /n Running environment...")
        observation = self.environment.run(parsed_output)
        step_results["observation"] = observation
        logging.info(f"Environment Observation: {observation}")
        return observation

    def generate_system_prompt(self):
        input_ = self.environment.inputs
        tools = self.environment.tools
        task = self.environment.task
        system_prompt = self.prompts["system_prompt"]
        messages = []
        for message in system_prompt:
            template = Template(message["content"])
            rendered_content = template.render(input=input_, tools=tools, task=task)
            messages.append({"role": message["role"], "content": rendered_content})
        return messages

    def generate_observation_prompt(self, action, observation):
        observation_prompt = self.prompts["observation_prompt"]
        messages = []
        for message in observation_prompt:
            template = Template(message["content"])
            rendered_content = template.render(action=action, observation=observation)
            messages.append({"role": message["role"], "content": rendered_content})
        return messages

    def run_slm(self, messages, step_results=None):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.stopping_criteria.reset(text)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_token_count = len(model_inputs.input_ids[0])
        if step_results:
            step_results["token_count"] = input_token_count
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=500,
            stopping_criteria=StoppingCriteriaList([self.stopping_criteria]),
            do_sample=True,
            temperature=0.7,
        )
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output

    def reset_memory(self):
        self.memory = []
        self.results = []

    def set_environment(self, new_environment):
        self.environment = new_environment

    def set_environment_task(self, new_task):
        self.environment.task = new_task


if __name__ == "__main__":
    dataset_path = RESEARCH_REPO_ROOT/"datasets"/"environment_datasets"/"clear_gta.csv"
    dataset = pd.read_csv(dataset_path)
    sample = dataset.iloc[0]
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    environment = StaticEnvironment(sample)
    agent = BaselineAgent(
        model_name=MODEL_NAME,
        environment=environment,
        prompts_file=RESEARCH_REPO_ROOT/"prompts"/"act_prompts_fs.yaml",
        max_steps=5,
    )
    agent.run()
