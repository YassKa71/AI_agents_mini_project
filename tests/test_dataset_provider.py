# tests/test_dataset_provider.py
import json
from pathlib import Path

import pandas as pd
import pytest

from research.environments.dataset_handling import dataset_provider as dp


@pytest.fixture
def sample_tool_meta() -> dict:
    return {
        "search_tool": {
            dp.DESCRIPTION: "Searches the web",
            dp.ARGUMENTS: {
                "query": {
                    dp.MATCH_TYPE: "exact",
                    dp.OPTIONAL: False,
                },
                "limit": {
                    dp.MATCH_TYPE: "numeric",
                    dp.OPTIONAL: True,
                },
            },
        },
        "calculator": {
            dp.DESCRIPTION: "Performs arithmetic",
            dp.ARGUMENTS: {
                "expression": {
                    dp.MATCH_TYPE: "exact",
                    dp.OPTIONAL: False,
                }
            },
        },
    }


@pytest.fixture
def sample_tasks_rows() -> list[dict]:
    return [
        {
            dp.TASK_ID: 1,
            dp.TASK: "Find the capital of France",
            dp.INPUTS: "No extra inputs",
            dp.TOOL_NAMES: json.dumps(["search_tool"]),
            dp.TOOL_CALLS: json.dumps(
                {
                    "search_tool": [
                        {
                            "arguments": {"query": "capital of France"},
                            "observation": "Paris",
                        }
                    ]
                }
            ),
            dp.FINAL_ANSWER: "Paris",
        },
        {
            dp.TASK_ID: 2,
            dp.TASK: "Compute 2 + 2",
            dp.INPUTS: "",
            dp.TOOL_NAMES: json.dumps(["calculator"]),
            dp.TOOL_CALLS: json.dumps(
                {
                    "calculator": [
                        {
                            "arguments": {"expression": "2 + 2"},
                            "observation": "4",
                        }
                    ]
                }
            ),
            dp.FINAL_ANSWER: "4",
        },
    ]


@pytest.fixture
def dataset_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Creates a fake ENVIRONMENT_DATASET_PATH and patches the module constant
    so DatasetProvider reads from the temp directory.
    """
    fake_root = tmp_path / "datasets"
    fake_root.mkdir(parents=True, exist_ok=True)
    return fake_root


@pytest.fixture
def dataset_dir(
    dataset_root: Path,
    sample_tool_meta: dict,
    sample_tasks_rows: list[dict],
) -> Path:
    dataset_name = "demo_dataset"
    dataset_path = dataset_root / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)

    # tool_meta.json
    tool_meta_path = dataset_path / dp.TOOL_META_PATH
    with tool_meta_path.open("w", encoding="utf-8") as f:
        json.dump(sample_tool_meta, f)

    # tasks.csv
    tasks_path = dataset_path / dp.TASKS_PATH
    df = pd.DataFrame(sample_tasks_rows)
    df.to_csv(tasks_path, index=False)

    return dataset_path


@pytest.fixture
def provider(dataset_dir: Path) -> dp.DatasetProvider:
    return dp.DatasetProvider(dataset_dir)


def test_init_loads_dataset_successfully(provider: dp.DatasetProvider) -> None:
    assert isinstance(provider.tool_meta, dict)
    assert isinstance(provider.tasks_df, pd.DataFrame)
    assert not provider.tasks_df.empty


def test_init_raises_if_dataset_directory_missing(dataset_root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Directory does not exist"):
        dp.DatasetProvider("missing_dataset")


def test_init_raises_if_tool_meta_missing(
    dataset_root: Path,
    sample_tasks_rows: list[dict],
) -> None:
    dataset_path = dataset_root / "missing_tool_meta"
    dataset_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(sample_tasks_rows).to_csv(dataset_path / dp.TASKS_PATH, index=False)

    with pytest.raises(FileNotFoundError, match=dp.TOOL_META_PATH):
        dp.DatasetProvider(dataset_path)


def test_init_raises_if_tasks_csv_missing(
    dataset_root: Path,
    sample_tool_meta: dict,
) -> None:
    dataset_path = dataset_root / "missing_tasks"
    dataset_path.mkdir(parents=True, exist_ok=True)

    with (dataset_path / dp.TOOL_META_PATH).open("w", encoding="utf-8") as f:
        json.dump(sample_tool_meta, f)

    with pytest.raises(FileNotFoundError, match=dp.TASKS_PATH):
        dp.DatasetProvider(dataset_path)


def test_get_task_tool_names(provider: dp.DatasetProvider) -> None:
    assert provider.get_task_tool_names(1) == ["search_tool"]


def test_get_task(provider: dp.DatasetProvider) -> None:
    assert provider.get_task(1) == "Find the capital of France"


def test_get_inputs(provider: dp.DatasetProvider) -> None:
    assert provider.get_inputs(1) == "No extra inputs"


def test_get_final_answer(provider: dp.DatasetProvider) -> None:
    assert provider.get_final_answer(1) == "Paris"


def test_get_matching_method(provider: dp.DatasetProvider) -> None:
    assert provider.get_matching_method("search_tool", "query") == "exact"


def test_get_argument_names(provider: dp.DatasetProvider) -> None:
    argument_names = provider.get_argument_names("search_tool")
    assert set(argument_names) == {"query", "limit"}


def test_get_all_tool_calls(provider: dp.DatasetProvider) -> None:
    result = provider.get_all_tool_calls(1)

    assert "search_tool" in result
    assert result["search_tool"][0]["arguments"]["query"] == "capital of France"
    assert result["search_tool"][0]["observation"] == "Paris"


def test_get_tool_calls(provider: dp.DatasetProvider) -> None:
    result = provider.get_tool_calls(2, "calculator")

    assert isinstance(result, list)
    assert result[0]["arguments"]["expression"] == "2 + 2"
    assert result[0]["observation"] == "4"


def test_get_tool_description(provider: dp.DatasetProvider) -> None:
    assert provider.get_tool_description("calculator") == "Performs arithmetic"


def test_get_tools_descriptions(provider: dp.DatasetProvider) -> None:
    result = provider.get_tools_descriptions(1)
    parsed = json.loads(result)

    assert parsed == ["Searches the web"]


def test_get_task_tool_names_raises_for_unknown_task(provider: dp.DatasetProvider) -> None:
    with pytest.raises(IndexError, match="task_id 999 not found"):
        provider.get_task_tool_names(999)


def test_get_task_raises_for_unknown_task(provider: dp.DatasetProvider) -> None:
    with pytest.raises(IndexError, match="task_id 999 not found"):
        provider.get_task(999)


def test_get_inputs_raises_for_unknown_task(provider: dp.DatasetProvider) -> None:
    with pytest.raises(IndexError, match="task_id 999 not found"):
        provider.get_inputs(999)


def test_get_final_answer_raises_for_unknown_task(provider: dp.DatasetProvider) -> None:
    with pytest.raises(IndexError, match="task_id 999 not found"):
        provider.get_final_answer(999)


def test_get_all_tool_calls_raises_for_unknown_task(provider: dp.DatasetProvider) -> None:
    with pytest.raises(IndexError, match="task_id 999 not found"):
        provider.get_all_tool_calls(999)


def test_get_tool_calls_raises_for_unknown_task(provider: dp.DatasetProvider) -> None:
    with pytest.raises(IndexError, match="task_id 999 not found"):
        provider.get_tool_calls(999, "search_tool")


def test_get_matching_method_raises_for_missing_tool(provider: dp.DatasetProvider) -> None:
    with pytest.raises(KeyError, match="Missing tool"):
        provider.get_matching_method("missing_tool", "query")


def test_get_matching_method_raises_for_missing_argument(provider: dp.DatasetProvider) -> None:
    with pytest.raises(KeyError, match="Missing argument"):
        provider.get_matching_method("search_tool", "missing_argument")


def test_get_argument_names_raises_for_missing_tool(provider: dp.DatasetProvider) -> None:
    with pytest.raises(KeyError, match="Missing tool"):
        provider.get_argument_names("missing_tool")


def test_get_tool_calls_raises_for_missing_tool_in_task(provider: dp.DatasetProvider) -> None:
    with pytest.raises(KeyError, match="Tool search_tool missing in tool_calls of task_id 2"):
        provider.get_tool_calls(2, "search_tool")


def test_get_tool_description_raises_for_missing_tool(provider: dp.DatasetProvider) -> None:
    with pytest.raises(KeyError, match="Missing tool"):
        provider.get_tool_description("missing_tool")


def test_missing_required_column_raises_keyerror(
    dataset_root: Path,
    sample_tool_meta: dict,
    sample_tasks_rows: list[dict],
) -> None:
    dataset_path = dataset_root / "missing_column_dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)

    with (dataset_path / dp.TOOL_META_PATH).open("w", encoding="utf-8") as f:
        json.dump(sample_tool_meta, f)

    rows = []
    for row in sample_tasks_rows:
        new_row = dict(row)
        new_row.pop(dp.FINAL_ANSWER)
        rows.append(new_row)

    pd.DataFrame(rows).to_csv(dataset_path / dp.TASKS_PATH, index=False)

    provider = dp.DatasetProvider(dataset_path)

    with pytest.raises(KeyError, match="misses a required column"):
        provider.get_final_answer(1)


def test_malformed_json_in_tasks_csv_raises(
    dataset_root: Path,
    sample_tool_meta: dict,
) -> None:
    dataset_path = dataset_root / "bad_json_dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)

    with (dataset_path / dp.TOOL_META_PATH).open("w", encoding="utf-8") as f:
        json.dump(sample_tool_meta, f)

    bad_df = pd.DataFrame(
        [
            {
                dp.TASK_ID: 1,
                dp.TASK: "Bad JSON example",
                dp.INPUTS: "",
                dp.TOOL_NAMES: '["search_tool"',  # malformed JSON
                dp.TOOL_CALLS: '{"search_tool": []}',
                dp.FINAL_ANSWER: "N/A",
            }
        ]
    )
    bad_df.to_csv(dataset_path / dp.TASKS_PATH, index=False)

    with pytest.raises(json.JSONDecodeError):
        dp.DatasetProvider(dataset_path)