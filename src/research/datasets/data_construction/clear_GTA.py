"""Creating a new benchmark based on GTA benchmark where we remove the
ambiguity about the content of the images.
Path of the new benchmark: research\datasets\environment_datasets\clear_gta.csv
"""

import json

import pandas as pd

from research.paths import RESEARCH_REPO_ROOT


def retrieve_image_descriptions(sample):
    """Retrieve image descriptions from a GTA benchmark sample

    Args:
        sample (json): sample from the gta dataset

    Returns:
        list: image descriptions
    """
    image_description = False
    dialogs = sample["dialogs"]
    for current_dialog in dialogs:
        if current_dialog.get("role") == "tool" and current_dialog.get("name") == "ImageDescription":
            image_description = True
            break
    if image_description:
        return [
            {
                "image_path": prev_dialog["tool_calls"][0]["function"]["arguments"]["image"],
                "image_description": current_dialog["content"]["content"],
            }
            for prev_dialog, current_dialog in zip(dialogs, dialogs[1:])
            if current_dialog.get("role") == "tool" and current_dialog.get("name") == "ImageDescription"
        ]
    else:
        return [
            {
                "image_path": file["path"],
                "image_description": "",
            }
            for file in sample["files"]
        ]


def extract_tools_final_answer(sample):
    """return filtered tools (removingImageDescription tool) and final answer from a sample

    Args:
        sample (json): sample from the gta dataset

    """
    final_answer = sample["gt_answer"]
    if isinstance(final_answer, dict):
        final_answer = final_answer["whitelist"]
    tools = sample["tools"]
    filtered_tools = [tool for tool in tools if tool["name"] != "ImageDescription"]
    return final_answer, filtered_tools


def create_dataset(path):
    """
    Create the clear GTA dataset. Should contain the following columns:
    - input: additional input/context to be added to the instruction
    - instruction: user query
    - tools: available tools
    - final answer
    - tool_calls: all needed tool calls to answer the query. Should be in the following format:
        {"tool_name":[{"arguments":{"argument1_name": ,...}, "observation":},...]}


    Args:
        path (Path): path to the GTA dataset

    Returns:
        DataFrame: Clear GTA dataset
    """
    # read json file
    with open(path) as file:
        data = json.load(file)
    len_data = len(data.keys())

    # create clear_gta
    clear_gta = {"input": [], "instruction": [], "tools": [], "final_answer": [], "tool_calls": []}

    for data_index in range(len_data):  # loop over all data samples
        sample = data[str(data_index)]
        # extract images descriptions and add them to input
        descriptions = retrieve_image_descriptions(sample)
        clear_gta["input"].append(descriptions)
        # add tools and final answer
        final_answer, filtered_tools = extract_tools_final_answer(sample)
        clear_gta["tools"].append(json.dumps(filtered_tools))
        clear_gta["final_answer"].append(final_answer)
        # add instruction and tool_calls
        instruction = ""
        tool_calls = {}
        dialogs = sample["dialogs"]
        for dialog_index in range(len(dialogs)):
            dialog = dialogs[dialog_index]
            dialog_role = dialog["role"]
            if dialog_role == "user":
                instruction = dialog["content"]
            elif dialog_role == "tool":
                tool_name = dialog["name"]
                if tool_name == "ImageDescription":
                    pass
                else:
                    observation = dialog["content"]["content"]
                    arguments = dialogs[dialog_index - 1]["tool_calls"][0]["function"]["arguments"]
                    tool_call = {"arguments": arguments, "observation": observation}
                    if tool_name in tool_calls:
                        tool_calls[tool_name].append(tool_call)
                    else:
                        tool_calls[tool_name] = []
                        tool_calls[tool_name].append(tool_call)
        if final_answer is None:
            final_tool_call = dialogs[-1]
            if final_tool_call["role"] == "tool":
                final_tool_call_name = final_tool_call["name"]
            elif final_tool_call["role"] == "assistant":
                final_tool_call_name = dialogs[-2]["name"]
            tool_calls[final_tool_call_name][-1]["observation"] = "success"
        clear_gta["tool_calls"].append(json.dumps(tool_calls))
        clear_gta["instruction"].append(instruction)

    # create df
    return pd.DataFrame(clear_gta)


if __name__ == "__main__":
    path = RESEARCH_REPO_ROOT / "datasets" / "environment_datasets" / "dataset.json"  # ADD path to the gta dataset
    clear_gta_df = create_dataset(path)
    output_path = RESEARCH_REPO_ROOT / "datasets" / "environment_datasets" / "clear_gta.csv"  # ADD output path
    clear_gta_df.to_csv(output_path)
