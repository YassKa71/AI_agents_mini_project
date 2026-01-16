"""
Simple AI agents Environment simulations for research purpose.
"""

import json
import logging

from rapidfuzz import fuzz

from research.paths import RESEARCH_REPO_ROOT

# Configure logging
logging.basicConfig(level=logging.INFO)


class StaticEnvironment:
    """ The environment is termed "static" because it does not execute external APIs
    or environment functions. Instead, it evaluates an AI agent’s responses by
    comparing them against the annotated ground-truth action sequences provided
    by the GTA benchmark. It uses either exact matching, fuzzy matching, or llm
    matching depending on the requested tool and based on the configuration in
    the file: research\datasets\environment_datasets\toolmeta.json.

    TODO:
    - Generalize this environment so it can support any AI-agent dataset that
    provides a reference (ground-truth) sequence of actions for each task."""
    def __init__(self, sample, slm, tool_meta=None, fuzz_treshold=25):
        logging.info("Environment Initialization...")
        if not tool_meta:
            tool_meta_path = RESEARCH_REPO_ROOT / "datasets" / "environment_datasets" / "toolmeta.json"
            with open(tool_meta_path) as f:
                tool_meta = json.load(f)
        self.TOOL_META = tool_meta
        self.SLM = slm
        self.task = sample.loc["instruction"]
        self.tools = json.loads(sample.loc["tools"])
        self.inputs = sample.loc["input"]
        self.outputs = []
        self.tool_calls = json.loads(sample.loc["tool_calls"])
        self.final_answer = sample.loc["final_answer"]
        self.OBSERVATION_TYPE_SUCCESS = "success"
        self.fuzz_treshold = fuzz_treshold
        self.wrong_answer_keywords = ["ValueError", "TypeError", "wrong direction"]
        logging.info(f"Environment successfully initialized. /n Task:{self.task} /n")

    def run(self, action):
        requested_tool_name = action["name"]
        requested_tool_arguments = action["arguments"]

        # if final answer check if success
        if requested_tool_name == "FinalAnswer":
            return self.check_final_answer(requested_tool_arguments)

        # return error message if requested tool not found
        selected_tool = self.search_tool(requested_tool_name)
        if not selected_tool:
            return f"ValueError: Requested Tool {requested_tool_name} not found in list of accessible tools."

        # check missing arguments
        for argument in selected_tool["inputs"]:
            if not argument["optional"] and not any(
                requested_argument["name"] == argument["name"] for requested_argument in requested_tool_arguments
            ):
                return f"TypeError: Missing argument {argument['name']}."

        # check if extra argument
        for requested_argument in requested_tool_arguments:
            found_argument = False
            for argument in selected_tool["inputs"]:
                if argument["name"] == requested_argument["name"]:
                    found_argument = True
            if not found_argument:
                return f"ValueError: Argument name {requested_argument['name']} not recognized by the tool."

        # run if present in tool_requests
        found_tool_call = self.search_tool_call(selected_tool, requested_tool_arguments)
        if found_tool_call is None:
            return "The results of your action will lead you in the wrong direction. You may need to try either other argument values or other tools."
        else:
            return (
                f"{found_tool_call['observation']}"
                if found_tool_call["observation"] is not None
                else "Action Done successfully."
            )

    def check_final_answer(self, requested_tool_arguments):
        if (
            len(requested_tool_arguments) > 0
            and "value" in requested_tool_arguments[0]
        ):
            correct_answer = self.llm_match("FinalAnswer", 0, requested_tool_arguments[0]["value"], self.final_answer)
            return self.OBSERVATION_TYPE_SUCCESS if correct_answer=="True" else "Incorrect final answer. Try again."
        else:
            return "Incorrect final answer. Try again."

    def llm_match(self, tool_name, argument_index, argument_value, valid_value):
        description = self.TOOL_META[tool_name]["description"]
        argument_name = self.TOOL_META[tool_name]["inputs"][argument_index]["name"]
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

        response = self.SLM.run(messages)
        return response

    def search_tool(self, tool_name):
        # return the tool with name tool_name
        selected_tool = None
        for tool in self.tools:
            if tool["name"] == tool_name:
                selected_tool = tool
                break
        return selected_tool

    def search_tool_call(self, selected_tool, requested_tool_arguments):
        # return the tool call matching requested arguments
        if selected_tool["name"] not in self.tool_calls:
            return None
        # search for the tool in tool_meta
        tool_meta = self.TOOL_META[selected_tool["name"]]

        # apply matching with argument value
        selected_tool_calls = self.tool_calls[selected_tool["name"]]
        for tool_call in selected_tool_calls:
            found_tool_call = None
            for argument_name, argument_value in tool_call["arguments"].items():
                found_argument = False
                for requested_argument in requested_tool_arguments:
                    if requested_argument["name"] == argument_name:
                        matching_method = "fuzzy_match"
                        index_argument = -1
                        for input_ in tool_meta["inputs"]:
                            if input_["name"] == argument_name:
                                matching_method = input_["match_type"]
                            index_argument += 1
                        found_argument = self.find_argument(matching_method, requested_argument, argument_value, selected_tool["name"], index_argument)
                        if not found_argument:
                            break

            if found_argument is True:
                found_tool_call = tool_call
                break
        return found_tool_call

    def find_argument(self, matching_method, requested_argument, argument_value, tool_name, index_argument):
        found_argument = False
        if matching_method == "fuzzy_match":
            if fuzz.partial_ratio(str(requested_argument["value"]), str(argument_value)) > self.fuzz_treshold:
                found_argument = True
            else:
                found_argument = False
        elif matching_method == "exact_match":
            if str(requested_argument["value"]) == str(argument_value):
                found_argument = True
        elif matching_method == "llm_match":
            llm_answer = self.llm_match(tool_name, index_argument ,requested_argument["value"], argument_value)
            if llm_answer == "True":
                found_argument = True
        else:
            found_argument = True
        return found_argument


""" TODO: To avoid magic strings we may need to implement the following classes:
class Tool:
    def __init__(self, name, description, arguments, outputs):
        self.name = name
        self.description = description
        self.arguments = arguments
        self.outputs = outputs

class Argument:
    def __init__(self, name, type_, optional=False, default=None, description=None):
        self.name = name
        self.type = type_
        self.optional = optional
        self.default = default
        self.description = description

And we may need to add methods to check if an inputed tool / argument match the tool / argument.
"""
