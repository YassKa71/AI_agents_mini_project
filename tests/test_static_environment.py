import pandas as pd
import pytest
import json

from research.environments.environment_simulation import (
    StaticEnvironment,
    RESPONSES,
    WARNING_STRATEGY,
    FINAL_ANSWER_TOOL,
    SUCCESS,
    PARSING_ERROR,
    INCORRECT_FINAL_ANSWER,
    MISSING_ARGUMENT,
    EXTRA_ARGUMENT,
    RIGHT_DIRECTION,
    WRONG_DIRECTION,
    FUZZY_MATCH,
    LLM_MATCH,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "task_id": [0],
            "task": ["Get weather for a city."],
            "tool_names": ['["weather_tool"]'],
            "input": ["City: Paris"],
            "tool_calls": [
                '{"weather_tool": [{"arguments": {"city": "Paris"}, "observation": "Sunny, 25°C"}]}'
            ],
            "final_answer": ["The weather in Paris is Sunny, 25°C"],
        }
    )


@pytest.fixture
def slm():
    class DummySLM:
        def run(self, messages):
            prompt = messages[0]["content"]
            if "Requested argument value: The weather in Paris is Sunny, 25°C" in prompt:
                return "True"
            if "Requested argument value: Paris" in prompt and "Valid argument value: Paris" in prompt:
                return "True"
            return "False"

    return DummySLM()


@pytest.fixture
def toolmeta():
    return {
        "weather_tool": {
            "description": """
                "name": "weather_tool",
                "description": "A weather tool.",
                "inputs": [
                {
                    "type": "text",
                    "name": "city",
                    "description": None,
                    "optional": False,
                    "default": None,
                    "filetype": None
                }
                ],
                "outputs": []""",
            "arguments": {
                "city": {
                    "type": str,
                    "optional": False,
                    "match_type": "exact_match",
                }
            },
        },
        "FinalAnswer": {
            "description": """
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
                "outputs": []""",
            "arguments": {
                "final_answer": {
                    "type": str,
                    "optional": False,
                    "match_type": "llm_match",
                }
            },
        },
    }


class FakeDatasetProvider:
    def __init__(self, sample_data, toolmeta):
        self.sample_data = sample_data
        self.toolmeta = toolmeta

    def get_argument_names(self, tool_name):
        return list(self.toolmeta[tool_name]["arguments"].keys())

    def is_argument_optional(self, tool_name, argument_name):
        return self.toolmeta[tool_name]["arguments"][argument_name]["optional"]

    def get_tool_calls(self, task_id, tool_name):
        raw = self.sample_data.loc[self.sample_data["task_id"] == task_id, "tool_calls"].iloc[0]
        parsed = json.loads(raw)
        return parsed.get(tool_name, [])

    def get_final_answer(self, task_id):
        return self.sample_data.loc[self.sample_data["task_id"] == task_id, "final_answer"].iloc[0]

    def get_task_tool_names(self, task_id):
        raw = self.sample_data.loc[self.sample_data["task_id"] == task_id, "tool_names"].iloc[0]
        return json.loads(raw)
    
    def get_task(self, task_id):
        raw = self.sample_data.loc[self.sample_data["task_id"] == task_id, "task"].iloc[0]
        return raw

    def get_matching_method(self, tool_name, argument_name):
        return self.toolmeta[tool_name]["arguments"][argument_name]["match_type"]

    def get_tool_description(self, tool_name):
        return self.toolmeta[tool_name]["description"]


@pytest.fixture
def data_provider(sample_data, toolmeta):
    return FakeDatasetProvider(sample_data, toolmeta)


@pytest.fixture
def env(slm, data_provider):
    return StaticEnvironment(
        slm=slm,
        data_provider=data_provider,
        current_task_id=0,
        fuzz_threshold=25,
        run_strategy=WARNING_STRATEGY,
    )


def test_init_raises_when_no_dataset_dir_and_no_provider(slm):
    with pytest.raises(ValueError, match="either 'dataset_dir' or 'data_provider' must be provided"):
        StaticEnvironment(slm=slm)


def test_exact_match(env):
    assert env.exact_match("Paris", "Paris") is True
    assert env.exact_match("Paris", "London") is False
    assert env.exact_match(123, "123") is True


def test_fuzzy_match(env):
    assert env.fuzzy_match("Parii", "Paris") is True
    assert env.fuzzy_match("Belop", "Paris") is False


def test_check_missing_argument(env):
    args = [{"name": "country", "value": "France"}]

    assert env.check_missing_argument("city", args)
    assert not env.check_missing_argument("country", args)


def test_check_extra_argument(env):
    tool_args = ["city"]

    assert env.check_extra_argument("country", tool_args)
    assert not env.check_extra_argument("city", tool_args)


def test_check_same_tool_call_exact(env):
    requested_arguments = [{"name": "city", "value": "Paris"}]
    tool_call = {"arguments": {"city": "Paris"}, "observation": "Sunny, 25°C"}

    assert env.check_same_tool_call("weather_tool", requested_arguments, tool_call)


def test_check_same_tool_call_exact_fail(env):
    requested_arguments = [{"name": "city", "value": "London"}]
    tool_call = {"arguments": {"city": "Paris"}, "observation": "Sunny, 25°C"}

    assert not env.check_same_tool_call("weather_tool", requested_arguments, tool_call)


def test_check_same_tool_call_fuzzy(slm, sample_data, toolmeta):
    toolmeta["weather_tool"]["arguments"]["city"]["match_type"] = FUZZY_MATCH

    env = StaticEnvironment(
        slm=slm,
        data_provider=FakeDatasetProvider(sample_data, toolmeta),
        current_task_id=0,
    )

    requested_arguments = [{"name": "city", "value": "Par"}]
    tool_call = {"arguments": {"city": "Paris"}, "observation": "Sunny, 25°C"}

    assert env.check_same_tool_call("weather_tool", requested_arguments, tool_call)


def test_check_same_tool_call_llm(slm, sample_data, toolmeta):
    toolmeta["weather_tool"]["arguments"]["city"]["match_type"] = LLM_MATCH

    env = StaticEnvironment(
        slm=slm,
        data_provider=FakeDatasetProvider(sample_data, toolmeta),
        current_task_id=0,
    )

    requested_arguments = [{"name": "city", "value": "Paris"}]
    tool_call = {"arguments": {"city": "Paris"}, "observation": "Sunny, 25°C"}

    assert env.check_same_tool_call("weather_tool", requested_arguments, tool_call)


def test_run_parsing_error(env, monkeypatch):
    module = __import__(StaticEnvironment.__module__, fromlist=["get_action"])

    def bad_parse(_):
        raise ValueError()

    monkeypatch.setattr(module, "get_action", bad_parse)

    result = env.run("bad output")

    assert result == RESPONSES[PARSING_ERROR]


def test_run_missing_argument(env, monkeypatch):
    module = __import__(StaticEnvironment.__module__, fromlist=["get_action"])

    monkeypatch.setattr(
        module,
        "get_action",
        lambda _: {
            "name": "weather_tool",
            "arguments": [],
        },
    )

    result = env.run("test")

    assert result["type"] == MISSING_ARGUMENT


def test_run_extra_argument(env, monkeypatch):
    module = __import__(StaticEnvironment.__module__, fromlist=["get_action"])

    monkeypatch.setattr(
        module,
        "get_action",
        lambda _: {
            "name": "weather_tool",
            "arguments": [
                {"name": "city", "value": "Paris"},
                {"name": "country", "value": "France"},
            ],
        },
    )

    result = env.run("test")

    assert result["type"] == EXTRA_ARGUMENT


def test_run_right_direction(env, monkeypatch):
    module = __import__(StaticEnvironment.__module__, fromlist=["get_action"])

    monkeypatch.setattr(
        module,
        "get_action",
        lambda _: {
            "name": "weather_tool",
            "arguments": [{"name": "city", "value": "Paris"}],
        },
    )

    result = env.run("test")

    assert result["type"] == RIGHT_DIRECTION
    assert result["message"] == "Sunny, 25°C"


def test_run_wrong_direction(env, monkeypatch):
    module = __import__(StaticEnvironment.__module__, fromlist=["get_action"])

    monkeypatch.setattr(
        module,
        "get_action",
        lambda _: {
            "name": "weather_tool",
            "arguments": [{"name": "city", "value": "London"}],
        },
    )

    result = env.run("test")

    assert result == RESPONSES[WRONG_DIRECTION]


def test_run_final_answer_success(env, monkeypatch):
    module = __import__(StaticEnvironment.__module__, fromlist=["get_action"])

    monkeypatch.setattr(
        module,
        "get_action",
        lambda _: {
            "name": FINAL_ANSWER_TOOL,
            "arguments": [
                {"name": "final_answer", "value": "The weather in Paris is Sunny, 25°C"}
            ],
        },
    )

    result = env.run("test")

    assert result == RESPONSES[SUCCESS]


def test_run_final_answer_incorrect(env, monkeypatch):
    module = __import__(StaticEnvironment.__module__, fromlist=["get_action"])

    monkeypatch.setattr(
        module,
        "get_action",
        lambda _: {
            "name": FINAL_ANSWER_TOOL,
            "arguments": [{"name": "final_answer", "value": "Wrong answer"}],
        },
    )

    result = env.run("test")

    assert result == RESPONSES[INCORRECT_FINAL_ANSWER]
