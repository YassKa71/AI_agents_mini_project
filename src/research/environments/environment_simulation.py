"""
Simple AI agents Environment simulations for research purpose.
"""

import logging

from rapidfuzz import fuzz
from pathlib import Path
from typing import overload

from research.environments.dataset_handling.dataset_provider import DatasetProvider
from research.utils.utils import SLM
from research.utils.utils import get_action

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants to avoid magic strings
# run strategies 
WARNING_STRATEGY = "warning message"
SLM_STRATEGY = "slm response"
# column or key names
TOOL_NAME = "name"
ARGUMENTS = "arguments"
MESSAGE = "message"
OBSERVATION = "observation"
ARGUMENT_VALUE = "value"
ARGUMENT_NAME = "name"
# tool names
FINAL_ANSWER_TOOL = "FinalAnswer"
# response types
PARSING_ERROR = "parsing_error"
WRONG_DIRECTION = "wrong_direction"
INCORRECT_FINAL_ANSWER = "incorrect_final_answer"
SUCCESS = "success"
MISSING_TOOL = "missing_tool"
MISSING_ARGUMENT = "missing_argument"
EXTRA_ARGUMENT = "extra_argument"
RIGHT_DIRECTION = "right_direction"
# matching methods
EXACT_MATCH = "exact_match"
FUZZY_MATCH = "fuzzy_match"
LLM_MATCH = "llm_match"
NO_MATCH = "no_match"


# Configuration of responses
RESPONSES = {
    PARSING_ERROR: {"status": "fail", "type": PARSING_ERROR, "message": "Error while parsing the output."},
    WRONG_DIRECTION: {"status": "success", "type": WRONG_DIRECTION, "message": "The results of your action will lead you in the wrong direction. You may need to try either other argument values or other tools."},
    INCORRECT_FINAL_ANSWER: {"status": "success", "type": INCORRECT_FINAL_ANSWER, "message": "Incorrect final answer. Try again."},
    SUCCESS: {"status": "success", "type": SUCCESS, "message": "Task completed successfully!"},
    MISSING_TOOL: {"status": "fail", "type": MISSING_TOOL, "message": "ValueError: Requested Tool {requested_tool_name} not found in list of accessible tools."},
    MISSING_ARGUMENT: {"status": "fail", "type": MISSING_ARGUMENT, "message": "TypeError: Missing argument {argument_name}."},
    EXTRA_ARGUMENT: {"status": "fail", "type": EXTRA_ARGUMENT, "message":"ValueError: Argument name {requested_argument_name} not recognized by the tool."},
    RIGHT_DIRECTION: {"status": "success", "type": RIGHT_DIRECTION, "message": lambda tool_call: f"{tool_call[OBSERVATION]}" if tool_call[OBSERVATION] is not None else "Action Done successfully."}
}

class StaticEnvironment:
    """ The environment is termed "static" because it does not execute external APIs
    or environment functions. Instead, it evaluates an AI agent’s responses by
    comparing them against an annotated ground-truth action sequences.
     
    This class requires as input a DataProvider object associated with a specific
    AI agent dataset (e.g., the GTA dataset) or the directory of the dataset.

    Evaluation is performed using one of the following methods: exact matching,
    fuzzy matching, or LLM-based matching. The method used depends on the
    requested tool and the configuration defined in the dataset's `toolmeta.json`
    file.
    """
    @overload
    def __init__(self, slm: SLM, dataset_dir: Path, data_provider: None = None, current_task_id: int = 0, fuzz_threshold: float = 25, run_strategy: str = WARNING_STRATEGY) -> None:...
    @overload
    def __init__(self, slm: SLM, dataset_dir: None = None, data_provider: DatasetProvider=..., current_task_id: int = 0, fuzz_threshold: float = 25, run_strategy: str = WARNING_STRATEGY) -> None:...

    def __init__(
        self, 
        slm: SLM,
        dataset_dir: Path|None = None,
        data_provider: DatasetProvider|None = None, 
        current_task_id: int = 0,
        fuzz_threshold: float = 25,
        run_strategy: str = WARNING_STRATEGY
    ) -> None:
        # raising errors or logging warnings
        if dataset_dir is None and data_provider is None:
            raise ValueError("Missing arguments: either 'dataset_dir' or 'data_provider' must be provided.")
        if dataset_dir is not None and data_provider is not None:
            logging.warning("Both 'dataset_dir' and 'data_provider' were provided. If they refer to "
        "different datasets, this may lead to inconsistent or incorrect results.")
        
        # initialization
        logging.info("Environment Initialization...")
        self.data_provider : DatasetProvider = (
            data_provider if data_provider is not None else DatasetProvider(dataset_dir)
        )
        self.slm : SLM = slm
        self.fuzz_threshold : float = fuzz_threshold
        self.current_task_id : int = current_task_id
        self.run_strategy : str = run_strategy

        logging.info(f"Environment successfully initialized. \n Current Task_id:{self.current_task_id} \n Task: {self.data_provider.get_task(self.current_task_id)}")

    def run(self, output: str) -> dict:

        # get action from ai agent output
        try:
            action = get_action(output)
        except ValueError:
            return RESPONSES.get(PARSING_ERROR)

        requested_tool_name = action[TOOL_NAME]
        requested_arguments = action[ARGUMENTS]

        # return response for final answer
        if requested_tool_name == FINAL_ANSWER_TOOL:
            if self.check_incorrect_final_answer(requested_arguments):
                return RESPONSES.get(INCORRECT_FINAL_ANSWER)
            else:
                return RESPONSES.get(SUCCESS)

        # return error message if requested tool not found
        if self.check_missing_tool(requested_tool_name):
            template = RESPONSES.get(MISSING_TOOL).copy()
            template[MESSAGE] = template[MESSAGE].format(requested_tool_name=requested_tool_name)
            return template
        
        tool_argument_names = self.data_provider.get_argument_names(requested_tool_name)

        # check missing arguments
        for argument_name in tool_argument_names:
            if not self.data_provider.is_argument_optional(requested_tool_name, argument_name
                ) and self.check_missing_argument(argument_name, requested_arguments):
                template = RESPONSES.get(MISSING_ARGUMENT).copy()
                template[MESSAGE] = template[MESSAGE].format(argument_name=argument_name)
                return template

        # check if extra argument
        for requested_argument in requested_arguments:
            if self.check_extra_argument(requested_argument[ARGUMENT_NAME], tool_argument_names):
                template = RESPONSES.get(EXTRA_ARGUMENT).copy()
                template[MESSAGE] = template[MESSAGE].format(requested_argument_name=requested_argument[ARGUMENT_NAME])
                return template

        # return observation if present in tool_calls
        for tool_call in self.data_provider.get_tool_calls(self.current_task_id, requested_tool_name):
            if self.check_same_tool_call(requested_tool_name, requested_arguments, tool_call):
                template = RESPONSES.get(RIGHT_DIRECTION).copy()
                template[MESSAGE] = template[MESSAGE](tool_call=tool_call)
                return template
        
        # return wrong direction message
        # TODO: return the wrong direction message only if self.run_startegy is WARNING_STRATEGY otherwise
        # if self.run_strategy is SLM_STRATEGY return a response generated by an slm
        return RESPONSES.get(WRONG_DIRECTION)
    
    def check_incorrect_final_answer(self, requested_arguments: list) -> bool:
        final_answer = self.data_provider.get_final_answer(self.current_task_id)
        if (
            len(requested_arguments) > 0
            and ARGUMENT_VALUE in requested_arguments[0]
            and ARGUMENT_NAME in requested_arguments[0]
        ):
            correct_answer = self.llm_match(FINAL_ANSWER_TOOL, requested_arguments[0][ARGUMENT_NAME], requested_arguments[0][ARGUMENT_VALUE], final_answer)
            return False if correct_answer=="True" else True
        else:
            return True
    
    def check_missing_tool(self, requested_tool_name: str) -> bool:
        # check if the requested tool name is hallucinated
        tool_names = self.data_provider.get_task_tool_names(self.current_task_id)
        return (requested_tool_name not in tool_names)
    
    def check_missing_argument(self, argument_name: str, requested_arguments: list) -> bool:
        # check if an argument name is missing in requested_arguments
        return all(argument_name != requested_argument[ARGUMENT_NAME] for requested_argument in requested_arguments)
    
    def check_extra_argument(self, requested_argument_name: str, tool_argument_names: list) -> bool:
        # check if the requested_argument_name is hallucinated
        return (requested_argument_name not in tool_argument_names)
    
    def check_same_tool_call(self, tool_name: str, requested_arguments: list, tool_call: dict) -> bool:
        for argument in requested_arguments:
            matching_type = self.data_provider.get_matching_method(tool_name, argument[ARGUMENT_NAME])
            if matching_type != NO_MATCH and not argument[ARGUMENT_NAME] in tool_call[ARGUMENTS]:
                return False
            else:
                if matching_type == EXACT_MATCH:
                    if not self.exact_match(argument[ARGUMENT_VALUE], tool_call[ARGUMENTS][argument[ARGUMENT_NAME]]):
                        return False
                elif matching_type == FUZZY_MATCH:
                    if not self.fuzzy_match(argument[ARGUMENT_VALUE], tool_call[ARGUMENTS][argument[ARGUMENT_NAME]]):
                        return False
                elif matching_type == NO_MATCH:
                    continue
                elif matching_type == LLM_MATCH:
                    match_result = self.llm_match(tool_name, argument[ARGUMENT_NAME], argument[ARGUMENT_VALUE], tool_call[ARGUMENTS][argument[ARGUMENT_NAME]])
                    if match_result != "True":
                        return False
                else:
                    logging.warning(f"Unrecognized matching method {matching_type} for argument {argument[ARGUMENT_NAME]} of tool {tool_name}. LLM matching was used instead.")
                    match_result = self.llm_match(tool_name, argument[ARGUMENT_NAME], argument[ARGUMENT_VALUE], tool_call[ARGUMENTS][argument[ARGUMENT_NAME]])
                    if match_result != "True":
                        return False
        return True

    def exact_match(self, requested_argument_value: str, argument_value: str) -> bool:
        return (str(requested_argument_value) == str(argument_value))

    def fuzzy_match(self, requested_argument_value: str, argument_value: str) -> bool:
        return (fuzz.partial_ratio(str(requested_argument_value), str(argument_value)) > self.fuzz_threshold)

    def llm_match(self, tool_name: str, argument_name: str, argument_value: str, valid_value: str) -> str:
        description = self.data_provider.get_tool_description(tool_name)
        prompt = f"""You're an AI expert verifying if the argument value of a tool_call is the same as a valid one.
        If the requested argument value is the same as the valid one return True. Otherwise return False.

        If the requested argument value lacks an important information that will change completely the tool output or lacks a precision compared to the valide one return False.

        The requested argument value doesn't have to be written the same way as the valid one but it has to contain the same necessary information to lead the tool to get the same output as the valid argument value does.

        Let's Begin. Here is the needed info:
        - Tool name: {tool_name}
        - Tool description: {description}
        - Argument name: {argument_name}
        - Requested argument value: {argument_value}
        - Valid argument value: {valid_value}

        Answer only with True or False nothing else.
        """

        messages= [
                {
                    "role": "user",
                    "content": prompt
                }
        ]

        response = self.slm.run(messages)
        return response
    
    def set_current_task_id(self, new_task_id: int):
        # set task id
        self.current_task_id = new_task_id

