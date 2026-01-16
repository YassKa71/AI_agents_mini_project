import pandas as pd
import pytest
import json

from research.ai_agents.environment.evaluation import StaticEnvironment
from research.utils.utils import SLM

@pytest.fixture
def sample_data():
    return pd.Series({
        "instruction": "Get weather for a city.",
        "tools": '[{"name": "weather_tool", "inputs": [{"name": "city", "optional": false}]}]',
        "input": "City: Paris",
        "tool_calls": '{"weather_tool": [{"arguments": {"city": "Paris"}, "observation": "Sunny, 25°C"}]}',
        "final_answer": "The weather in Paris is Sunny, 25°C",
    })

@pytest.fixture
def slm():
    return SLM("HuggingFaceTB/SmolLM2-135M-Instruct")

@pytest.fixture
def toolmeta():
    return ({
        "weather_tool": {
            "name": "weather_tool",
            "description": "A weather tool.",
            "inputs": [
            {
                "type": "text",
                "name": "name",
                "description": None,
                "optional": False,
                "default": None,
                "filetype": None,
                "match_type": "exact_match"
            }
            ],
            "outputs": []
        },
        "FinalAnswer": {
            "name": "FinalAnswer",
            "description": "This tool returns the final answer to the user.",
            "inputs": [
            {
                "type": "text",
                "name": "final_answer",
                "description": "The answer that will be sent to the user",
                "optional": "False",
                "default": "None",
                "filetype": "None",
                "match_type": "llm_match"
            }
            ],
            "outputs": []
        },
    })

def test_successful_tool_call(sample_data, slm, toolmeta):
    env = StaticEnvironment(sample_data, slm, tool_meta=toolmeta)
    action = {"name": "weather_tool", "arguments": [{"name": "city", "value": "Paris"}]}
    result = env.run(action)
    print(result)
    assert result == "Sunny, 25°C"


def test_missing_argument(sample_data, slm, toolmeta):
    env = StaticEnvironment(sample_data, slm, tool_meta=toolmeta)
    action = {
        "name": "weather_tool",
        "arguments": [],  # missing required "city"
    }
    result = env.run(action)
    assert "Missing argument city" in result


def test_unknown_tool(sample_data, slm, toolmeta):
    env = StaticEnvironment(sample_data, slm, tool_meta=toolmeta)
    action = {"name": "unknown_tool", "arguments": []}
    result = env.run(action)
    assert "Requested Tool unknown_tool not found" in result


def test_extra_argument(sample_data, slm, toolmeta):
    env = StaticEnvironment(sample_data, slm, tool_meta=toolmeta)
    action = {
        "name": "weather_tool",
        "arguments": [
            {"name": "city", "value": "Paris"},
            {"name": "country", "value": "France"},  # extra
        ],
    }
    result = env.run(action)
    assert "Argument name country not recognized" in result


def test_wrong_argument_value(sample_data, slm, toolmeta):
    env = StaticEnvironment(sample_data, slm, tool_meta=toolmeta)
    action = {"name": "weather_tool", "arguments": [{"name": "city", "value": "Lyon"}]}
    result = env.run(action)
    assert "wrong direction" in result


def test_correct_final_answer(sample_data, slm, toolmeta):
    env = StaticEnvironment(sample_data, slm, tool_meta=toolmeta)
    action = {"name": "FinalAnswer", "arguments": [{"name": "value", "value": "The weather in Paris is Sunny, 25°C"}]}
    result = env.run(action)
    assert result == "success"


def test_incorrect_final_answer(sample_data, slm, toolmeta):
    env = StaticEnvironment(sample_data, slm, tool_meta=toolmeta)
    action = {"name": "FinalAnswer", "arguments": [{"name": "value", "value": "Rainy in London"}]}
    result = env.run(action)
    assert result == "Incorrect final answer. Try again."
