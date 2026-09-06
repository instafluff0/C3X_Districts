"""Faster indexed variant of the proven static CIVBLP package reader.

Large packages can contain hundreds of thousands of allocation records.  The
original structural reader intentionally favors simple exhaustive inference;
this adapter keeps the same validation rules while caching reflected types and
restricting allocation-table candidates to the only possible 40-byte phase.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from Renderer.tools.asset_compiler import civblp_probe
from Renderer.tools.asset_compiler import clutter_blp_extractor


ALLOCATION_BYTES = 40
SEARCH_BYTES = 8192


def find_indexed_allocation_table(package: bytes) -> tuple[int, list[dict[str, object]]]:
    marker = package.rfind(civblp_probe.ENTRY_MAP_TYPE)
    if marker < 0:
        raise ValueError("Static package has no reflected entry-map type marker")
    search_start = marker + len(civblp_probe.ENTRY_MAP_TYPE)
    search_end = min(len(package) - ALLOCATION_BYTES, search_start + SEARCH_BYTES)
    candidates: list[tuple[int, list[dict[str, object]]]] = []
    for start in range(search_start, search_end + 1):
        if package[start] != 0 or package[start + 2 : start + 6] != b"\0" * 4:
            continue
        head = civblp_probe.unpack_allocation(package, start)
        if (
            head["parent_pointer"] != 0
            or head["target_offset"] != 0
            or not head["size"]
            or not head["element_count"]
            or head["size"] % head["element_count"]
        ):
            continue
        parsed = clutter_blp_extractor._parse_allocation_candidate(package, start)
        if parsed is not None:
            candidates.append((start, parsed))
    if not candidates:
        raise ValueError("Could not locate the phased static-package allocation table")
    largest = max(len(candidate[1]) for candidate in candidates)
    winners = [candidate for candidate in candidates if len(candidate[1]) == largest]
    if len(winners) != 1:
        raise ValueError("Static-package allocation-table selection is ambiguous")
    return winners[0]


class IndexedStaticPackage(clutter_blp_extractor.StaticPackage):
    def __init__(
        self,
        source: Path,
        target_string: str,
        *,
        allow_declared_size_mismatch: bool = False,
        minimum_temp_support: int = 3,
    ) -> None:
        self.source = source
        self.data = source.read_bytes()
        self.header = civblp_probe.parse_file_header(
            self.data[:28],
            len(self.data),
            allow_declared_size_mismatch=allow_declared_size_mismatch,
        )
        self.package_file_offset = self.header["package_data"]["offset"]
        self.big_data_file_offset = self.header["big_data"]["offset"]
        self.package = self.data[self.package_file_offset : self.big_data_file_offset]
        self.table_offset, self.allocations = find_indexed_allocation_table(self.package)
        temp_base, self.temp_evidence = civblp_probe.infer_temp_stripe_base(
            self.package, self.allocations, minimum_temp_support
        )
        package_base, self.target_char_pointer, self.package_evidence = self._infer_package_base(
            temp_base, target_string
        )
        self.stripe_bases = {0: package_base, 1: temp_base}
        self._type_cache: dict[int, str | None] = {}

    def _infer_package_base(self, temp_base: int, target: str) -> tuple[int, int, list[str]]:
        encoded = target.encode("ascii") + b"\0"
        target_offsets = civblp_probe.raw_occurrences(self.package, encoded)
        if len(target_offsets) != 1:
            raise ValueError(f"Expected target name once in package data, found {len(target_offsets)}")
        target_offset = target_offsets[0]
        temp_bases = {1: temp_base}
        type_cache: dict[int, str | None] = {}

        def allocation_type(pointer: int) -> str | None:
            type_pointer = self.allocations[pointer - 1]["type_pointer"]
            if type_pointer not in type_cache:
                type_cache[type_pointer] = civblp_probe.resolve_type_name(
                    type_pointer,
                    self.package,
                    self.allocations,
                    temp_bases,
                )
            return type_cache[type_pointer]

        char_allocations = [
            (pointer, allocation)
            for pointer, allocation in enumerate(self.allocations, 1)
            if allocation["stripe"] == 0
            and allocation["parent_pointer"] == 0
            and allocation_type(pointer) == civblp_probe.CHAR_TYPE
        ]
        candidate_bases: set[int] = set()
        for _pointer, allocation in char_allocations:
            deltas = []
            if allocation["size"] == len(encoded):
                deltas.append(0)
            if allocation["size"] == len(encoded) + 8:
                deltas.append(8)
            for delta in deltas:
                candidate = target_offset - allocation["target_offset"] - delta
                resolved = civblp_probe.read_allocated_string(
                    self.package,
                    candidate + allocation["target_offset"],
                    allocation["size"],
                )
                if resolved is not None and resolved[0] == target:
                    candidate_bases.add(candidate)
        if not candidate_bases:
            raise ValueError("Could not infer package stripe base from the selected target")

        scores: Counter[int] = Counter()
        for candidate in candidate_bases:
            for _pointer, allocation in char_allocations:
                if civblp_probe.read_allocated_string(
                    self.package,
                    candidate + allocation["target_offset"],
                    allocation["size"],
                ) is not None:
                    scores[candidate] += 1
        package_base, score = scores.most_common(1)[0]
        if score < 2 or list(scores.values()).count(score) != 1:
            raise ValueError("Package stripe-base inference was not unique")
        pointers = []
        for pointer, allocation in char_allocations:
            resolved = civblp_probe.read_allocated_string(
                self.package,
                package_base + allocation["target_offset"],
                allocation["size"],
            )
            if resolved is not None and resolved[0] == target:
                pointers.append(pointer)
        if len(pointers) != 1:
            raise ValueError("Selected target does not resolve to one char allocation")
        return package_base, pointers[0], [f"cached reflected-type inference; score={score}"]

    def type_name(self, pointer: int) -> str | None:
        if pointer < 1 or pointer > len(self.allocations):
            return None
        type_pointer = self.allocations[pointer - 1]["type_pointer"]
        if type_pointer not in self._type_cache:
            self._type_cache[type_pointer] = civblp_probe.resolve_type_name(
                type_pointer,
                self.package,
                self.allocations,
                self.stripe_bases,
            )
        return self._type_cache[type_pointer]
