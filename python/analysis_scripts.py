from __future__ import annotations
import duckdb

import argparse
from pathlib import Path
from textwrap import dedent

DEFAULT_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
METRIC_COLUMNS = (
	"goal_distance_delta",
	"progress",
	"approach_bonus",
	"trajectory_alignment_reward",
	"speed_penalty",
	"living_penalty",
)

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Aggregate a step-level training CSV into one row per episode."
	)
	parser.add_argument(
		"input_csv",
		help="CSV filename or path for the step log to aggregate.",
	)
	parser.add_argument(
		"-o",
		"--output",
		help="Optional output CSV path. Defaults to processed_<input name> beside the input.",
	)
	return parser.parse_args()

def resolve_input_path(raw_input: str) -> Path:
	input_path = Path(raw_input).expanduser()
	if input_path.is_file():
		return input_path.resolve()

	logs_path = DEFAULT_LOGS_DIR / input_path
	if logs_path.is_file():
		return logs_path.resolve()

	raise FileNotFoundError(
		f"Could not find input CSV '{raw_input}'. Checked the current directory and {DEFAULT_LOGS_DIR}."
	)

def resolve_output_path(input_path: Path, raw_output: str | None) -> Path:
	if raw_output:
		return Path(raw_output).expanduser().resolve()
	return input_path.with_name(f"processed_{input_path.name}")

def sql_literal(value: str) -> str:
	return value.replace("'", "''")

def build_aggregate_query(input_path: Path) -> str:
	escaped_path = sql_literal(input_path.as_posix())
	metric_averages = ",\n".join(
		f"    AVG({column}) AS {column}" for column in METRIC_COLUMNS
	)
	return dedent(
		f"""
		SELECT
		    episode,
		{metric_averages}
		FROM read_csv_auto('{escaped_path}')
		GROUP BY episode
		ORDER BY episode
		"""
	).strip()

def build_summary_query(input_path: Path) -> str:
	escaped_path = sql_literal(input_path.as_posix())
	return dedent(
		f"""
		SELECT
		    COUNT(*) AS step_rows,
		    COUNT(DISTINCT episode) AS episode_rows,
		    MIN(episode) AS first_episode,
		    MAX(episode) AS last_episode
		FROM read_csv_auto('{escaped_path}')
		"""
	).strip()

def main() -> None:
	args = parse_args()
	input_path = resolve_input_path(args.input_csv)
	output_path = resolve_output_path(input_path, args.output)

	with duckdb.connect() as connection:
		summary_row = connection.sql(build_summary_query(input_path)).fetchone()
		if summary_row is None:
			raise RuntimeError(f"No rows found in {input_path}")
		step_rows, episode_rows, first_episode, last_episode = summary_row
		connection.sql(build_aggregate_query(input_path)).write_csv(output_path.as_posix())

	print(
		f"Wrote {output_path.name} with {episode_rows} episode rows "
		f"(episodes {first_episode}..{last_episode}) from {step_rows} step rows."
	)

if __name__ == "__main__":
	main()
