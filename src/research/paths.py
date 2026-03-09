from pathlib import Path

RESEARCH_REPO_ROOT = Path(__file__).resolve().parents[0]
ENVIRONMENT_DATASET_PATH = RESEARCH_REPO_ROOT / "environments" / "datasets"

if __name__ == "__main__":
    print(RESEARCH_REPO_ROOT)
