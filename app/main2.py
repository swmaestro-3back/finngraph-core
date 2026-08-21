import argparse
import asyncio
import uuid
from pathlib import Path

from langchain_core.tracers.langchain import wait_for_all_tracers

from app.crud import upsert_triplets
from app.scripts.seed_db import seed
from app.graph.workflow import GraphRunner

# Default location of the news datasets (e.g. tests/data/samsung).
DATA_ROOT = Path(__file__).resolve().parent.parent / "tests" / "data"


def resolve_dir(name: str) -> Path:
    """Resolve a dataset folder name to a directory.

    Accepts either a bare folder name (looked up under tests/data) or a path.
    """
    candidates = [DATA_ROOT / name, Path(name)]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Dataset directory not found: {name}")


def load_texts(directory: Path) -> list[tuple[str, str]]:
    """Read every ``*.py`` file in the directory and return (filename, TEXT) pairs.

    Each data file is a plain module defining a single ``TEXT`` string, so it is
    executed in an isolated namespace and the ``TEXT`` binding is pulled out.
    Files sorted by name to keep the run order deterministic.
    """
    texts: list[tuple[str, str]] = []
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        namespace: dict = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
        text = namespace.get("TEXT")
        if not text:
            print(f"[skip] {path.name}: no TEXT defined")
            continue
        texts.append((path.name, text))
    return texts


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the graph over every news file in a dataset folder."
    )
    parser.add_argument(
        "folder",
        help="Dataset folder name under tests/data (e.g. samsung), or a path",
    )
    args = parser.parse_args()

    directory = resolve_dir(args.folder)
    texts = load_texts(directory)
    if not texts:
        print(f"No news files found in {directory}")
        return

    print(f"Found {len(texts)} news files in {directory}")

    # Ensure database is seeded before running
    await seed()

    runner = GraphRunner()
    total_triplets = 0
    failures: list[tuple[str, Exception]] = []
    try:
        # Run one file at a time; the next TEXT only starts once the previous
        # one has been fully processed and persisted.
        for index, (filename, text) in enumerate(texts, start=1):
            print(f"\n[{index}/{len(texts)}] {filename}")
            try:
                news_id = str(uuid.uuid4())
                final_state = await runner.ainvoke(news_id, text)

                # Upsert to neo4j
                await upsert_triplets(news_id, final_state["triplets"])

                count = len(final_state["triplets"])
                total_triplets += count
                print(f"    done. {count} triplets extracted.")
            except Exception as exc:  # keep going so one bad file doesn't stop the run
                failures.append((filename, exc))
                print(f"    failed: {exc!r}")

        print(
            f"\nCompleted {len(texts) - len(failures)}/{len(texts)} files. "
            f"{total_triplets} triplets extracted in total."
        )
        if failures:
            print("Failed files:")
            for filename, exc in failures:
                print(f"  - {filename}: {exc!r}")
    finally:
        # LangSmith SDK uploads traces asynchronously via background threads.
        # Short-lived scripts may terminate before root workflow run events (end/output)
        # are fully flushed, leaving the top-level trace output empty in LangSmith.
        # Ensure all pending traces are flushed before process exit.
        wait_for_all_tracers()


if __name__ == "__main__":
    asyncio.run(main())
