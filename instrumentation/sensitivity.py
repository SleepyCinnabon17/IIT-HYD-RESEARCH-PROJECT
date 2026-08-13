"""Post-hoc gate sensitivity analysis using fixed detector/TTA outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline import classify_risk

CONFIGURATIONS = {
    "V=0.13 (C=0.35)": (0.35, 0.13),
    "V=0.15 (C=0.35)": (0.35, 0.15),
    "V=0.17 (C=0.35)": (0.35, 0.17),
    "C=0.30 (V=0.15)": (0.30, 0.15),
    "C=0.35 (V=0.15)": (0.35, 0.15),
    "C=0.40 (V=0.15)": (0.40, 0.15),
}


def risk_for(
    item: dict[str, Any], confidence_threshold: float, variance_threshold: float
) -> str:
    return classify_risk(
        item["detected"],
        item["confidence"],
        item["variance"],
        confidence_threshold,
        variance_threshold,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("evidence/expanded/results.json")
    )
    parser.add_argument("--output", type=Path, default=Path("SENSITIVITY.md"))
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))

    labels: dict[str, dict[str, str]] = {}
    counts: dict[str, Counter[str]] = {name: Counter() for name in CONFIGURATIONS}
    for item in results:
        labels[item["image"]] = {}
        for name, (confidence_threshold, variance_threshold) in CONFIGURATIONS.items():
            risk = risk_for(item, confidence_threshold, variance_threshold)
            labels[item["image"]][name] = risk
            counts[name][risk] += 1

    columns = list(CONFIGURATIONS)
    lines = [
        "# Threshold Sensitivity",
        "",
        "This is an isolated, post-hoc analysis. It does not edit `pipeline.py`, rerun the VLM, or alter the locked Pass 5 evidence. Detector and five-pass TTA outputs are fixed; only the deterministic risk classifier is recomputed because confidence thresholds do not affect detector inference.",
        "",
        "> **Threshold chronology:** standard deviation `0.15` was the initial Pass 4/early Pass 5 value. Pass 5 tuning changed and locked the shipped value at `0.03` before authoritative acceptance; Pass 6 did not change it. The `0.13/0.15/0.17` and `0.30/0.35/0.40` grid below is the exact separately requested sensitivity reference, with the other parameter held at `0.15` or `0.35` as specified. No production threshold was changed by this analysis.",
        "",
        "## Per-Image Risk Labels",
        "",
        "| Image | " + " | ".join(columns) + " |",
        "|---|" + "|".join("---" for _ in columns) + "|",
    ]
    for item in results:
        lines.append(
            f"| `{item['image']}` | "
            + " | ".join(labels[item["image"]][name] for name in columns)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Risk Counts",
            "",
            "| Configuration | Low | Medium | High | N/A |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in columns:
        counter = counts[name]
        lines.append(
            f"| {name} | {counter['Low']} | {counter['Medium']} | {counter['High']} | {counter['N/A']} |"
        )

    reference = labels
    variance_changes = sum(
        len({reference[image][name] for name in columns[:3]}) > 1 for image in reference
    )
    confidence_changes = sum(
        len({reference[image][name] for name in columns[3:]}) > 1 for image in reference
    )
    shipped_to_requested_changes = sum(
        item["hallucination_risk"] != reference[item["image"]]["V=0.15 (C=0.35)"]
        for item in results
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"Across {len(results)} images, **{variance_changes}** change risk label "
                "somewhere over variance thresholds 0.13–0.17 and "
                f"**{confidence_changes}** change somewhere over confidence thresholds "
                "0.30–0.40. This indicates how many samples lie near the requested gate "
                "boundary; unchanged samples are stable within this narrow reference "
                "range. These counts characterize this curated small sample only and do "
                "not establish universal calibration. Separately, "
                f"**{shipped_to_requested_changes} of {len(results)}** labels differ "
                "between the shipped variance threshold 0.03 results and the requested "
                "0.15 reference. Therefore, internal stability of the requested grid "
                "must not be interpreted as insensitivity to the shipped threshold choice."
            ),
            "",
        ]
    )
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "images": len(results),
                "variance_label_changes": variance_changes,
                "confidence_label_changes": confidence_changes,
                "shipped_0_03_to_requested_0_15_changes": shipped_to_requested_changes,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
