"""
All simple paradigms of AI agents with a queue-structured memory system.
"""

import logging
import time
import yaml

from jinja2 import Template
from transformers import StoppingCriteriaList
from pathlib import Path

from research.utils.utils import ActionBlockStoppingCriteria, SLM, get_action
from research.environments.environment_simulation import StaticEnvironment, SUCCESS, PARSING_ERROR
from research.paths import RESEARCH_REPO_ROOT, ENVIRONMENT_DATASET_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)


class BaselineAgent:
    """
    The group of paradigms represented by this class are characterized by:
    - A queue memory system where information is appended over time
    - No planning process
    - Support for different prompt files which must include the following prompts: system_prompt,
        observation_prompt, parsing_error_prompt
    """

    def __init__(self, slm: SLM, environment: StaticEnvironment, prompts_file: str|Path, max_steps: int = 50) -> None:
        self.environment : StaticEnvironment = environment
        self.memory : list = []
        self.max_steps : int = max_steps
        self.results : list = []
        self.slm : SLM = slm
        with open(prompts_file) as file:
            self.prompts : dict = yaml.safe_load(file)
            required_keys = {"system_prompt", "observation_prompt", "parsing_error_prompt", "reformatting_prompt"}
            missing = required_keys - self.prompts.keys()
            if missing:
                raise ValueError(f"Missing prompt keys in {prompts_file}: {sorted(missing)}")
        self.stopping_criteria : ActionBlockStoppingCriteria = ActionBlockStoppingCriteria(self.slm.tokenizer)

    def run(self) -> None:
        """Main agent loop"""

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

            # Run SLM
            output = self._run_llm_step(step_results)

            # Run Environment
            logging.info("Running environment...")
            _, type_, observation = self.run_environment(output, start_time, step_results)

            # if task fulfilled stop the loop
            if type_ == SUCCESS:
                solved = self._handle_success(step_results)

            # if parsing error reformat and try again
            elif type_ == PARSING_ERROR:
                solved = self._handle_parsing_error(
                    output=output,
                    start_time=start_time,
                    step_results=step_results,
                )
                if not solved:
                    current_step += 1

            # if any other type of observation generate observation prompt and continue
            else:
                self._handle_observation(
                    output=output,
                    observation=observation,
                    step_results=step_results,
                )
                current_step += 1


    def _run_llm_step(self, step_results: dict) -> str:
        """Run the language model and record its output."""

        logging.info("Running the llm...")
        output = self.run_slm(self.memory, step_results=step_results)
        logging.info(f"llm output: {output}")

        # store llm output for debugging / analysis
        step_results["llm_output"] = output
        return output


    def _handle_success(self, step_results: dict) -> bool:
        """Handle successful completion of the task."""

        logging.info("Task Completed !!!")
        self.results.append(step_results)
        return True


    def _handle_parsing_error(
        self,
        output: str,
        start_time: float,
        step_results: dict,
    ) -> bool:
        """
        Handle parsing errors.

        First attempt: reformat the output and try running the environment again.
        If it still fails, store the error and prompt the model to fix formatting.
        """

        # use prompt for reformatting
        new_output = self.reformat(output)

        _, type_, observation = self.run_environment(new_output, start_time, step_results)

        # if task fulfilled after reformat stop the loop
        if type_ == SUCCESS:
            return self._handle_success(step_results)

        # if parsing still fails after reformat
        elif type_ == PARSING_ERROR:
            # adding duration to step_results
            duration = time.time() - start_time
            step_results["duration"] = duration

            # using parsing_error_prompt as observation
            reformat_prompt = self.generate_parsing_error_prompt(output)
            self.memory += reformat_prompt

            logging.warning("Error while parsing the output.")

            step_results["observation"] = "parsing error"
            self.results.append(step_results)

            return False

        # if reformat worked but environment returned another observation
        else:
            self._handle_observation(
                output=new_output,
                observation=observation,
                step_results=step_results,
            )
            return False


    def _handle_observation(
        self,
        output: str,
        observation: str,
        step_results: dict,
    ) -> None:
        """
        Handle tool/environment observations by generating the next prompt
        and appending it to memory.
        """

        tool_observation_prompt = self.generate_observation_prompt(
            get_action(output),
            observation,
        )

        # update agent memory with observation
        self.memory += tool_observation_prompt

        # store results for the step
        self.results.append(step_results)

    def reformat(self, output: str) -> str:
        # run the slm to correct the format of an output
        messages = []
        reformatting_prompt = self.prompts["reformatting_prompt"]
        for message in reformatting_prompt:
            template = Template(message["content"])
            rendered_content = template.render(output=output)
            messages.append({"role": message["role"], "content": rendered_content})
        new_output = self.run_slm(messages)
        return new_output

    def run_environment(self, output: str, start_time: float, step_results: dict) -> tuple[str, str, str]:
        # adding duration to step_results
        duration = time.time() - start_time
        step_results["duration"] = duration
        response = self.environment.run(output)
        status, type_, observation = response["status"], response["type"], response["message"]
        step_results["observation"] = response["message"]
        logging.info(f"Environment Response: {response}")
        return status, type_, observation

    def generate_system_prompt(self) -> list:
        inputs = self.environment.data_provider.get_inputs(self.environment.current_task_id)
        tool_names = self.environment.data_provider.get_task_tool_names(self.environment.current_task_id)
        task = self.environment.data_provider.get_task(self.environment.current_task_id)
        system_prompt = self.prompts["system_prompt"]
        messages = []
        for message in system_prompt:
            template = Template(message["content"])
            rendered_content = template.render(input=inputs, tools=tool_names, task=task)
            messages.append({"role": message["role"], "content": rendered_content})
        return messages
    
    def generate_parsing_error_prompt(self, output: str) -> list:
        parsing_error_prompt = self.prompts["parsing_error_prompt"]
        messages = []
        for message in parsing_error_prompt:
            template = Template(message["content"])
            rendered_content = template.render(output=output)
            messages.append({"role": message["role"], "content": rendered_content})
        return messages

    def generate_observation_prompt(self, action: dict, observation: str) -> list:
        observation_prompt = self.prompts["observation_prompt"]
        messages = []
        for message in observation_prompt:
            template = Template(message["content"])
            rendered_content = template.render(action=action, observation=observation)
            messages.append({"role": message["role"], "content": rendered_content})
        return messages

    def run_slm(self, messages: list, step_results: dict|None = None, temperature: float = 0.7, max_new_tokens: float = 500) -> str:
        text = self.slm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.stopping_criteria.reset(text)
        model_inputs = self.slm.tokenizer([text], return_tensors="pt").to(self.slm.model.device)
        input_token_count = len(model_inputs.input_ids[0])
        if step_results:
            step_results["token_count"] = input_token_count
        generated_ids = self.slm.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=StoppingCriteriaList([self.stopping_criteria]),
            temperature=temperature,
        )
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        output = self.slm.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output

    def reset_memory(self) -> None:
        self.memory = []
        self.results = []

    def set_environment(self, new_environment: StaticEnvironment) -> None:
        self.environment = new_environment

    def set_environment_task_id(self, new_task_id: int) -> None:
        self.environment.set_current_task_id(new_task_id)


if __name__ == "__main__":
    dataset_dir = ENVIRONMENT_DATASET_PATH / "GTA_dataset"
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    slm = SLM(MODEL_NAME)
    environment = StaticEnvironment(slm, dataset_dir)
    agent = BaselineAgent(
        slm=slm,
        environment=environment,
        prompts_file=RESEARCH_REPO_ROOT/"ai_agents"/"mono_agents"/"prompts"/"ReAct_prompts_fs.yaml",
        max_steps=3,
    )
    agent.run()
