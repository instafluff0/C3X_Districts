"""Reproduce the cross-owner transfer-function boundary without choosing a policy."""

import json
import math
from pathlib import Path


def witness():
    return {
        "schema": "c3x.lab_v2.linear_boundary_witness.v1",
        "input": "two black and two white opaque pixels reduced to one pixel",
        "frozen_encoded_byte_box": (0 + 255 + 0 + 255) // 4,
        "gamma_2_2_linear_light": round(255 * pow(0.5, 1 / 2.2)),
        "srgb_linear_light": round(255 * (1.055 * pow(0.5, 1 / 2.4) - 0.055)),
        "conclusion": "The frozen byte-box result cannot simultaneously be a linear-light reduction. A new attachment and Q6 transfer/exposure contract are required; this witness does not select either transform.",
    }


if __name__ == "__main__":
    print(json.dumps(witness(), indent=2))
