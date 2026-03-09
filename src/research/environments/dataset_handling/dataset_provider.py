"""
DatasetProvider provides access to information contained in AI agent
benchmarks and datasets.

Given a dataset (e.g., GTABenchmark) containing tool definitions,
descriptions, tasks, expected outputs, and related metadata, this
class acts as a retrieval layer for other components.

It allows querying the dataset to obtain relevant information such as:
- tool descriptions
- metadata associated with tools
- information about specific tool arguments
- task definitions and expected answers

Currently, DatasetProvider expects an input directory with the following files
during initialization:

1. A CSV file named "tasks.csv" containing a list of tasks and related information with
   the following columns:
   - task_id
   - task
   - inputs: helpful inputs for solving the task, such as
     images and their descriptions
   - tool_names: names of tools the agent is allowed to use
   - tool_calls: list of correct tool calls with observations
     to solve the task (useful for simulating an environment)
   - final_answer

2. A JSON file named "tool_meta.json" containing tool descriptions and metadata.

See the documentation for more details.

TODO: Generalize the class to adapt to new dataset formats.

"""
import json
import pandas as pd

from pathlib import Path


# Constants to avoid magic strings
# paths
TOOL_META_PATH = "tool_meta.json"
TASKS_PATH = "tasks.csv"
# tasks file columns' name
TASK_ID = "task_id"
TOOL_NAMES = "tool_names"
TOOL_CALLS = "tool_calls"
FINAL_ANSWER = "final_answer"
TASK = "task"
INPUTS = "inputs"
# tool meta file keys' name
ARGUMENTS = "arguments"
DESCRIPTION = "description"
MATCH_TYPE = "match_type"
OPTIONAL = "optional"

class DatasetProvider:
    """
    Provides access to AI agent dataset and tool metadata.
    """
    def __init__(self, dataset_directory: Path|str) -> None:

        # initialize directory path
        self.dataset_directory : Path = Path(dataset_directory)
        if not self.dataset_directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.dataset_directory}")
        
        # initialize required files
        self.tool_meta : dict = self._load_tool_meta()
        self.tasks_df : pd.DataFrame = self._load_tasks_df()
    
    def _load_tool_meta(self) -> dict:
        # load tool meta file as json
        tool_meta_path = self.dataset_directory / TOOL_META_PATH

        # raise error if file not found
        if not tool_meta_path.exists():
            raise FileNotFoundError(f"File {TOOL_META_PATH} not found in directory {self.dataset_directory}")

        with open(tool_meta_path, 'r') as tool_meta_file:
            return json.load(tool_meta_file)

    # TODO: json.loads may not work if columns are in Python list literal format. Solve this issue.    
    def _load_tasks_df(self) -> pd.DataFrame:
        # load tasks file as pandas DataFrame
        tasks_path = self.dataset_directory / TASKS_PATH

        # raise error if file not found
        if not tasks_path.exists():
            raise FileNotFoundError(f"File {TASKS_PATH} not found in directory {self.dataset_directory}")

        return pd.read_csv(
            tasks_path,
            converters={
                TOOL_NAMES: json.loads,
                TOOL_CALLS: json.loads
            }
        )
    
    def get_task_tool_names(self, task_id: int) -> list:
        try:
            tool_names = self.tasks_df.loc[self.tasks_df[TASK_ID] == task_id, TOOL_NAMES].iat[0]
            return tool_names
        except KeyError as e:
            raise KeyError("The tasks file misses a required column.") from e
        except IndexError as e:
            raise IndexError(f"task_id {task_id} not found in tasks file") from e
    
    def get_task(self, task_id: int) -> str:
        try:
            task = self.tasks_df.loc[self.tasks_df[TASK_ID] == task_id, TASK].iat[0]
            return task
        except KeyError as e:
            raise KeyError("The tasks file misses a required column.") from e
        except IndexError as e:
            raise IndexError(f"task_id {task_id} not found in tasks file") from e
    
    def get_inputs(self, task_id: int) -> str:
        try:
            inputs = self.tasks_df.loc[self.tasks_df[TASK_ID] == task_id, INPUTS].iat[0]
            return inputs
        except KeyError as e:
            raise KeyError("The tasks file misses a required column.") from e
        except IndexError as e:
            raise IndexError(f"task_id {task_id} not found in tasks file") from e

    def get_final_answer(self, task_id: int) -> str:
        try:
            final_answer = self.tasks_df.loc[self.tasks_df[TASK_ID] == task_id, FINAL_ANSWER].iat[0]
            return final_answer
        except KeyError as e:
            raise KeyError("The tasks file misses a required column.") from e
        except IndexError as e:
            raise IndexError(f"task_id {task_id} not found in tasks file") from e
    
    def is_argument_optional(self, tool_name: str, argument_name: str) -> bool:
        try:
            return self.tool_meta[tool_name][ARGUMENTS][argument_name][OPTIONAL]
        except KeyError as e:
            if e.args[0] == tool_name:
                raise KeyError(f" Missing tool {e.args[0]} in tool_meta file.") from e
            elif e.args[0] == argument_name:
                raise KeyError(f" Missing argument {e.args[0]} for tool {tool_name} in tool_meta file.") from e
            else:
                raise KeyError(f" Missing required key {e.args[0]} in tool_meta file.") from e

    def get_matching_method(self, tool_name: str, argument_name: str) -> str:
        try:
            return self.tool_meta[tool_name][ARGUMENTS][argument_name][MATCH_TYPE]
        except KeyError as e:
            if e.args[0] == tool_name:
                raise KeyError(f" Missing tool {e.args[0]} in tool_meta file.") from e
            elif e.args[0] == argument_name:
                raise KeyError(f" Missing argument {e.args[0]} for tool {tool_name} in tool_meta file.") from e
            else:
                raise KeyError(f" Missing required key {e.args[0]} in tool_meta file.") from e

    def get_argument_names(self, tool_name: str) -> list:
        try:
            arguments = self.tool_meta[tool_name][ARGUMENTS]
            return list(arguments.keys())
        except KeyError as e:
            if e.args[0] == tool_name:
                raise KeyError(f" Missing tool {e.args[0]} in tool_meta file.") from e
            else:
                raise KeyError(f" Missing required key {e.args[0]} in tool_meta file.") from e

    def get_all_tool_calls(self, task_id: int) -> dict:
        try:
            all_tool_calls = self.tasks_df.loc[self.tasks_df[TASK_ID] == task_id, TOOL_CALLS].iat[0]
            return all_tool_calls
        except KeyError as e:
            raise KeyError("The tasks file misses a required column.") from e
        except IndexError as e:
            raise IndexError(f"task_id {task_id} not found in tasks file") from e

    def get_tool_calls(self, task_id: int, tool_name: str) -> list:
        try:
            all_tool_calls = self.tasks_df.loc[self.tasks_df[TASK_ID] == task_id, TOOL_CALLS].iat[0]
            return all_tool_calls[tool_name]
        except KeyError as e:
            if e.args[0] == tool_name:
                raise KeyError(f"Tool {tool_name} missing in tool_calls of task_id {task_id}.") from e
            else:
                raise KeyError("The tasks file misses a required column.") from e
        except IndexError as e:
            raise IndexError(f"task_id {task_id} not found in tasks file") from e
    
    def get_tool_description(self, tool_name: str) -> str:
        try:
            return self.tool_meta[tool_name][DESCRIPTION]
        except KeyError as e:
            if e.args[0] == tool_name:
                raise KeyError(f" Missing tool {e.args[0]} in tool_meta file.") from e
            else:
                raise KeyError(f" Missing required key {e.args[0]} in tool_meta file.") from e

    def get_tools_descriptions(self, task_id: int) -> str:
        # get tool_names from tasks file
        
        tool_names = self.get_task_tool_names(task_id)
        # return descriptions
        descriptions = [self.get_tool_description(tool_name) for tool_name in tool_names]
        return json.dumps(descriptions)
