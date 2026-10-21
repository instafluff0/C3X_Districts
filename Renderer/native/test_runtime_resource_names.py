"""The Renderer64 runtime keys no behaviour on resource or unit names.

Civ III scenarios bring their own resources and units under names the renderer
cannot predict, so what a resource looks like belongs in pack metadata. Name
checks silently broke even our own content: when compositions began placing
the derived school "fish.few", fish lost the underwater rendering that a check
for the binding "fish" gave them. The runtime then still loaded a fixed list of
primary animation bindings ("horses", "cattle", ...), animated any resource
whose lowercase name contained a binding name, and drew a legacy static group
chosen the same way ("horses", "iron", "uranium", ...), with
`std::strcmp(group_name, "fish")` turning off its shadow. Packs now carry all of
it: composition aliases, the animation pack's "aliases" and each binding's
"medium".

This test walks the Renderer64 DLL's sources (the .cpp files on
BUILD_RENDERER64.bat's /LD line and their quoted includes) and fails on any
string literal that is, or has a path segment that is, a resource or unit name:
the vanilla Conquests rosters and every name, alias, profile and subject that
the resource profiles know, plus the static builder's group names. Comments
are not code. The few literals that merely share such a word are listed with
their exact counts below, so a new one still fails.
"""
import json
import re
import unittest
from collections import Counter

from Renderer.lab.platform import ROOT

INVENTORY = ROOT / "Renderer/inventory"
BUILD = ROOT / "Renderer/native/BUILD_RENDERER64.bat"
# Civ III and Civ VI identifier prefixes that wrap a name (PRTO_Worker).
PREFIXES = ("prto_", "resource_", "good_")
# Literals that share a name without keying game content on it, with counts.
ALLOWED = {
    # The executable module named in diagnostic traces.
    ("Renderer/native/c3x_renderer.cpp", "game"): 1,
    ("Renderer/native/native_callsite_diagnostic.h", "game"): 1,
    # The exported sandbox benchmark scene's fixed cast (c3x_sandbox_*), not game tiles.
    ("Renderer/sandbox/direct_units.h", "PRTO_Warrior"): 1,
    ("Renderer/sandbox/direct_units.h", "PRTO_Archer"): 1,
    ("Renderer/sandbox/direct_units.h", "PRTO_Worker"): 1,
    ("Renderer/sandbox/direct_units.h", "PRTO_Spearman"): 1,
    ("Renderer/sandbox/direct_units.h", "PRTO_Scout"): 1,
    ("Renderer/sandbox/direct_units.h", "Spearman"): 1,
    ("Renderer/sandbox/direct_units.h", "Scout"): 1,
}


def vocabulary():
    names = set()
    for roster in ("vanilla_conquests_to_civ6_resources.json", "vanilla_conquests_to_civ6_units.json"):
        names.update(row["civ3_name"] for row in json.loads((INVENTORY / roster).read_text())["mappings"])
    profiles = json.loads((INVENTORY / "resource_composition_profiles.json").read_text())
    names.update(profiles["bindings"])
    names.update(alias for aliases in profiles["bindings"].values() for alias in aliases)
    for key, profile in profiles["resources"].items():
        names.add(key)
        names.update(profile.get("subjects", ()))
        names.update([profile["subject"]] if profile.get("subject") else ())
    from Renderer.tools.asset_compiler.build_resource_runtime import SELECTIONS
    names.update(name for name, *_ in SELECTIONS)
    words = {name.lower() for name in names}
    return words | {word[:-1] for word in words if word.endswith("s") and len(word) > 4}


def dll_sources():
    """The DLL's translation units and every file they include by quoted path."""
    line = next(text for text in BUILD.read_text().splitlines() if " /LD " in text)
    pending = [(BUILD.parent / name.replace("\\", "/")).resolve() for name in re.findall(r"\S+\.cpp", line)]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for include in re.findall(r'^\s*#\s*include\s*"([^"]+)"', path.read_text(errors="replace"), re.M):
            pending.append((path.parent / include).resolve())
    return sorted(seen)


def literals(text):
    """(line, contents) of each string literal, skipping comments and character literals."""
    found, index, line = [], 0, 1
    while index < len(text):
        char = text[index]
        if text.startswith("//", index):
            end = text.find("\n", index)
            index = len(text) if end < 0 else end
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = len(text) if end < 0 else end + 2
        elif char == "R" and text.startswith('R"', index) and not (text[index - 1:index].isalnum() or
                                                                    text[index - 1:index] == "_"):
            opening = text.index("(", index)
            delimiter = text[index + 2:opening]
            closing = text.index(")" + delimiter + '"', opening)
            found.append((line, text[opening + 1:closing]))
            end = closing + len(delimiter) + 2
        elif char == "'" and text[index - 1:index].isdigit() and text[index + 1:index + 2].isalnum():
            end = index + 1   # a digit separator (1'000)
        elif char in "\"'":
            end = index + 1
            while end < len(text) and text[end] != char:
                end += 2 if text[end] == "\\" else 1
            if char == '"':
                found.append((line, text[index + 1:end]))
            end += 1
        else:
            end = index + 1
        line += text.count("\n", index, end)
        index = end
    return found


def names_in(literal, words):
    lowered = literal.lower()
    for segment in [lowered] + re.split(r"[/.:\\]", lowered):
        for candidate in [segment] + [segment[len(p):] for p in PREFIXES if segment.startswith(p)]:
            if candidate in words:
                return True
    return False


def name_literals(sources=None):
    """Counter of (file, literal) for literals naming a resource or unit, with their lines."""
    words, counts, lines = vocabulary(), Counter(), {}
    for path in sources or dll_sources():
        relative = path.relative_to(ROOT).as_posix()
        for line, literal in literals(path.read_text(errors="replace")):
            if names_in(literal, words):
                counts[relative, literal] += 1
                lines.setdefault((relative, literal), []).append(line)
    return counts, lines


class RuntimeResourceNameTests(unittest.TestCase):
    maxDiff = None

    def test_runtime_sources_name_no_resource_or_unit(self):
        sources = dll_sources()
        self.assertIn(ROOT / "Renderer/native/c3x_renderer.cpp", sources)
        counts, lines = name_literals(sources)
        unexpected = [f"{file}:{lines[file, literal]} \"{literal}\""
                      for (file, literal), count in sorted(counts.items())
                      if count != ALLOWED.get((file, literal), 0)]
        self.assertEqual(unexpected, [], "resource/unit names belong in pack metadata, not runtime code")

    def test_scanner_sees_the_retired_name_checks(self):
        words = vocabulary()
        for literal in ("horses", "Fish", "dye", "fish.few", "animated/whales", "PRTO_Galley", "horse"):
            self.assertTrue(names_in(literal, words), literal)
        for literal in ("worker-complete", "game_thread_sampler.h", "image-batch-transport", "medium", "water"):
            self.assertFalse(names_in(literal, words), literal)
        source = 'char const* a = "x"; // "fish"\n/* "iron" */ bool f = std::strcmp(g, "fish") == 0; char c = \'"\';'
        self.assertEqual(literals(source), [(1, "x"), (2, "fish")])


if __name__ == "__main__":
    unittest.main()
