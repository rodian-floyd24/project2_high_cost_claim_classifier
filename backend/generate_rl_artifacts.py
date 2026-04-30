from __future__ import annotations

from pathlib import Path

from rl.q_learning import save_metadata, save_q_table, train_q_table


ARTIFACT_DIR = Path(__file__).resolve().parent / "model_artifacts"
Q_TABLE_PATH = ARTIFACT_DIR / "q_table.json"
RL_METADATA_PATH = ARTIFACT_DIR / "rl_metadata.json"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    q_table = train_q_table()
    save_q_table(Q_TABLE_PATH, q_table)
    save_metadata(RL_METADATA_PATH)
    print(f"Saved {Q_TABLE_PATH}")
    print(f"Saved {RL_METADATA_PATH}")


if __name__ == "__main__":
    main()
