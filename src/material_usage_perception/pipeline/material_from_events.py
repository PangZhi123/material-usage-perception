"""Estimate and aggregate material usage from a vehicle-event JSON document."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from material_usage_perception.domain import VehicleVisitEvent, VisitOutcome
from material_usage_perception.estimation import (
    MaterialEstimator,
    MaterialQuantityConfig,
    MaterialUsageAccumulator,
)


def estimate_event_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config: MaterialQuantityConfig | None = None,
) -> dict:
    raw_events = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(raw_events, list):
        raise ValueError("vehicle event document must be a JSON list")
    estimator = MaterialEstimator(config or MaterialQuantityConfig())
    accumulator = MaterialUsageAccumulator()
    records = []
    for raw in raw_events:
        event = VehicleVisitEvent(
            **{**raw, "outcome": VisitOutcome(raw["outcome"])}
        )
        record = estimator.estimate(event)
        accumulator.add(record)
        records.append(record)
    result = {
        "records": [asdict(record) for record in records],
        "summaries": [asdict(summary) for summary in accumulator.summaries()],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate concrete use from vehicle events.")
    parser.add_argument("--input-events", required=True)
    parser.add_argument("--output-material", required=True)
    parser.add_argument("--default-capacity-m3", type=float, default=12.0)
    parser.add_argument("--default-load-factor", type=float, default=1.0)
    parser.add_argument("--default-unload-factor", type=float, default=0.95)
    args = parser.parse_args()
    result = estimate_event_file(
        args.input_events,
        args.output_material,
        config=MaterialQuantityConfig(
            args.default_capacity_m3,
            args.default_load_factor,
            args.default_unload_factor,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
