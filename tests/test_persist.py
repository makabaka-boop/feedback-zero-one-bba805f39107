"""Tests for the persist backend, runnable with ``python3 -m unittest``.

The suite covers validation/rejection behaviour, the reference square
example from the specification, edge cases (empty input, coincident
points, tetrahedra) and a randomised differential check against the
independent verifier in ``tools/verify.py``.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from persist.errors import InputError  # noqa: E402
from persist.pipeline import dumps, run, run_file  # noqa: E402

import verify  # noqa: E402


def point(identifier, xyz):
    return {"id": identifier, "xyz": xyz}


def document(points, prime=2, cutoff="2/1", queries=("0/1",)):
    return {
        "points": points,
        "prime": prime,
        "cutoff": cutoff,
        "queries": list(queries),
    }


SQUARE = document(
    [
        point("d", ["0/1", "1/1", "0/1"]),
        point("b", ["1/1", "0/1", "0/1"]),
        point("a", ["0/1", "0/1", "0/1"]),
        point("c", ["1/1", "1/1", "0/1"]),
    ],
    prime=2,
    cutoff="2/1",
    queries=["1/1", "2/1"],
)


class ReferenceExample(unittest.TestCase):
    def test_unit_square(self):
        output = run(json.dumps(SQUARE))
        self.assertEqual(
            output["bars"],
            [
                [0, "0/1", "1/1"],
                [0, "0/1", "1/1"],
                [0, "0/1", "1/1"],
                [0, "0/1", None],
                [1, "1/1", "2/1"],
            ],
        )
        self.assertEqual(
            output["answers"],
            [
                {
                    "betti": [1, 1, 0],
                    "cycles": [
                        [
                            [["a", "b"], 1],
                            [["a", "d"], 1],
                            [["b", "c"], 1],
                            [["c", "d"], 1],
                        ]
                    ],
                },
                {"betti": [1, 0, 0], "cycles": []},
            ],
        )

    def test_input_order_is_irrelevant(self):
        shuffled = document(
            list(reversed(SQUARE["points"])),
            prime=2,
            cutoff="2/1",
            queries=["2/1", "1/1"],
        )
        a = run(json.dumps(SQUARE))
        b = run(json.dumps(shuffled))
        self.assertEqual(a["bars"], b["bars"])
        self.assertEqual(a["answers"][0], b["answers"][1])
        self.assertEqual(a["answers"][1], b["answers"][0])


class EdgeCases(unittest.TestCase):
    def test_empty_points_and_queries(self):
        output = run(json.dumps(document([], queries=[])))
        self.assertEqual(output, {"bars": [], "answers": []})

    def test_single_point(self):
        output = run(json.dumps(document([point("a", ["0/1", "0/1", "0/1"])], prime=3, cutoff="5/2")))
        self.assertEqual(output["bars"], [[0, "0/1", None]])
        self.assertEqual(output["answers"][0]["betti"], [1, 0, 0])

    def test_coincident_points_are_distinct(self):
        output = run(
            json.dumps(
                document(
                    [
                        point("a", ["0/1", "0/1", "0/1"]),
                        point("b", ["0/1", "0/1", "0/1"]),
                        point("c", ["0/1", "0/1", "0/1"]),
                    ],
                    cutoff="0/1",
                )
            )
        )
        # All three vertices enter at scale 0 and coincide, so they stay
        # connected: one component alive for the whole window.
        self.assertEqual(output["bars"], [[0, "0/1", None]])
        self.assertEqual(output["answers"][0]["betti"], [1, 0, 0])

    def test_octahedron_has_h2(self):
        points = [
            point("p", ["1/1", "0/1", "0/1"]),
            point("n", ["-1/1", "0/1", "0/1"]),
            point("q", ["0/1", "1/1", "0/1"]),
            point("r", ["0/1", "-1/1", "0/1"]),
            point("s", ["0/1", "0/1", "1/1"]),
            point("t", ["0/1", "0/1", "-1/1"]),
        ]
        output = run(json.dumps(document(points, cutoff="4/1", queries=["2/1", "4/1"])))
        self.assertIn([2, "2/1", "4/1"], output["bars"])
        self.assertEqual(output["answers"][0]["betti"], [1, 0, 1])
        self.assertEqual(output["answers"][1]["betti"], [1, 0, 0])

    def test_zero_length_bars_omitted_and_duplicates_kept(self):
        # Three edges of length 4/1 kill three components at the same
        # scale; the resulting four coexisting component bars are kept
        # individually.
        points = [point(c, [f"{i}/1", "0/1", "0/1"]) for i, c in enumerate("abcd")]
        output = run(json.dumps(document(points, cutoff="9/1", queries=["1/1"])))
        deaths = [bar for bar in output["bars"] if bar[0] == 0]
        self.assertEqual(len(deaths), 4)
        self.assertEqual(sum(1 for bar in deaths if bar[2] is None), 1)
        self.assertTrue(all(bar[1] < bar[2] for bar in deaths if bar[2] is not None))

    def test_scale_is_squared_distance(self):
        # Distance 2 in x gives squared distance 4; an edge must not
        # appear for a cutoff of 3, leaving two components for the window.
        doc = document(
            [point("a", ["0/1", "0/1", "0/1"]), point("b", ["2/1", "0/1", "0/1"])],
            cutoff="3/1",
            queries=["3/1"],
        )
        self.assertEqual(run(json.dumps(doc))["bars"], [[0, "0/1", None], [0, "0/1", None]])
        doc["cutoff"] = "4/1"
        self.assertEqual(
            run(json.dumps(doc))["bars"],
            [[0, "0/1", "4/1"], [0, "0/1", None]],
        )

    def test_unreduced_rationals_are_normalised(self):
        doc = document(
            [point("a", ["0/1", "0/1", "0/1"]), point("b", ["12/8", "0/1", "0/1"])],
            cutoff="36/16",
            queries=["9/4"],
        )
        output = run(json.dumps(doc))
        self.assertEqual(output["bars"], [[0, "0/1", "9/4"], [0, "0/1", None]])
        self.assertEqual(output["answers"][0]["betti"], [1, 0, 0])

    def test_negative_coordinates(self):
        doc = document(
            [point("a", ["-3/2", "0/1", "0/1"]), point("b", ["0/1", "0/1", "0/1"])],
            cutoff="9/4",
            queries=["9/4"],
        )
        self.assertEqual(run(json.dumps(doc))["bars"], [[0, "0/1", "9/4"], [0, "0/1", None]])


class Rejections(unittest.TestCase):
    def reject(self, payload, path, index=None):
        with self.assertRaises(InputError) as context:
            run(json.dumps(payload))
        self.assertEqual(context.exception.path, path)
        if index is not None:
            self.assertEqual(context.exception.index, index)

    def reject_text(self, text, path):
        with self.assertRaises(InputError) as context:
            run(text)
        self.assertEqual(context.exception.path, path)

    def test_syntax_error_at_root(self):
        self.reject_text('{"points": [],}', "$")
        self.reject_text('', "$")
        self.reject_text('{"points": []', "$")
        self.reject_text('NaN', "$")
        self.reject_text('{"points": [], "prime": 2, "cutoff": "1/1", "queries": [], "x": Infinity}', "$")

    def test_duplicate_member_reported_with_path(self):
        self.reject_text(
            '{"points": [{"id": "a", "id": "b", "xyz": ["0/1","0/1","0/1"]}],'
            ' "prime": 2, "cutoff": "1/1", "queries": []}',
            "$.points[0].id",
        )

    def test_unknown_member(self):
        self.reject({"points": [], "prime": 2, "cutoff": "1/1", "queries": [], "extra": 1}, "$.extra")
        doc = document([point("a", ["0/1", "0/1", "0/1"])])
        doc["points"][0]["w"] = 1
        self.reject(doc, "$.points[0].w", 0)

    def test_missing_member(self):
        self.reject({"points": [], "prime": 2, "cutoff": "1/1"}, "$.queries")
        doc = document([{"id": "a", "xyz": ["0/1", "0/1"]}])
        self.reject(doc, "$.points[0].xyz", 0)

    def test_duplicate_id(self):
        doc = document([point("a", ["0/1", "0/1", "0/1"]), point("a", ["1/1", "0/1", "0/1"])])
        self.reject(doc, "$.points[1].id", 1)

    def test_bad_ids(self):
        for bad in ("", "a b", "a-b", "é", "a\n", "a_b"):
            self.reject(document([point(bad, ["0/1", "0/1", "0/1"])]), "$.points[0].id", 0)

    def test_bad_prime(self):
        for bad in (True, False, "2", 2.0, 1, 4, 11, None):
            doc = document([], prime=bad)
            self.reject(doc, "$.prime")

    def test_bad_rationals(self):
        cases = [
            ("1", "$.cutoff"),
            ("1/0", "$.cutoff"),
            ("1/-2", "$.cutoff"),
            ("1.5", "$.cutoff"),
            ("abc", "$.cutoff"),
            (1, "$.cutoff"),
            ("1/2/3", "$.cutoff"),
            (" 1/2", "$.cutoff"),
        ]
        for value, path in cases:
            doc = document([], cutoff=value)
            self.reject(doc, path)

    def test_bad_xyz(self):
        doc = document([point("a", ["0/1", "0/1"])])
        self.reject(doc, "$.points[0].xyz", 0)
        doc = document([point("a", ["0/1", "0/1", "0/1", "0/1"])])
        self.reject(doc, "$.points[0].xyz", 0)
        doc = document([point("a", ["0/1", True, "0/1"])])
        self.reject(doc, "$.points[0].xyz[1]", 1)
        doc = document([point("a", ["0/1", "1/0", "0/1"])])
        self.reject(doc, "$.points[0].xyz[1]", 1)

    def test_query_out_of_range(self):
        doc = document([], cutoff="1/1", queries=["3/2"])
        self.reject(doc, "$.queries[0]", 0)
        doc = document([], cutoff="1/1", queries=["-1/2"])
        self.reject(doc, "$.queries[0]", 0)

    def test_top_level_not_object(self):
        self.reject_text("[]", "$")


class OutputPreservation(unittest.TestCase):
    def test_failure_keeps_old_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "in.json")
            target = os.path.join(directory, "out.json")
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("OLD")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write('{"points": []}')
            with self.assertRaises(InputError):
                run_file(source, target)
            with open(target, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "OLD")

    def test_success_replaces_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "in.json")
            target = os.path.join(directory, "out.json")
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("OLD")
            with open(source, "w", encoding="utf-8") as handle:
                json.dump(SQUARE, handle)
            run_file(source, target)
            with open(target, encoding="utf-8") as handle:
                output = json.load(handle)
            self.assertEqual(len(output["answers"]), 2)
            self.assertEqual(
                sorted(os.listdir(directory)), ["in.json", "out.json"]
            )

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "in.json")
            target = os.path.join(directory, "out.json")
            with open(source, "w", encoding="utf-8") as handle:
                json.dump(SQUARE, handle)
            result = subprocess.run(
                [sys.executable, os.path.join(ROOT, "persist.py"), source, target],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(source, "w", encoding="utf-8") as handle:
                json.dump(
                    {"points": [], "prime": 2, "cutoff": "1/1", "queries": ["2/1"]},
                    handle,
                )
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("OLD")
            result = subprocess.run(
                [sys.executable, os.path.join(ROOT, "persist.py"), source, target],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("$.queries", result.stderr)
            with open(target, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "OLD")


class Differential(unittest.TestCase):
    """Randomised cross-check against the independent verifier."""

    def check_document(self, doc):
        text = json.dumps(doc)
        output = run(text)
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "in.json")
            target = os.path.join(directory, "out.json")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write(text)
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(dumps(output))
            verify.check(source, target)

    def test_random_documents(self):
        random.seed(20240517)
        for _ in range(40):
            count = random.randint(0, 10)
            points = [
                point(
                    random.choice("abAB") + str(i),
                    [f"{random.randint(-3, 3)}/{random.randint(1, 3)}" for _ in range(3)],
                )
                for i in range(count)
            ]
            cutoff = f"{random.randint(0, 16)}/{random.randint(1, 3)}"
            limit = Fraction(cutoff)
            prime = random.choice([2, 3, 5, 7])
            candidates = {
                Fraction(random.randint(0, 16), random.randint(1, 3))
                for _ in range(random.randint(0, 4))
            }
            queries = [
                f"{value.numerator}/{value.denominator}"
                for value in sorted(value for value in candidates if 0 <= value <= limit)
            ]
            self.check_document(
                document(points, prime=prime, cutoff=cutoff, queries=queries)
            )

    def test_dense_grid_documents(self):
        random.seed(271828)
        grid = list(itertools.product(range(-1, 2), repeat=3))
        for _ in range(15):
            random.shuffle(grid)
            count = random.randint(4, 8)
            points = [
                point(f"P{i}", [f"{coordinate}/1" for coordinate in grid[i]])
                for i in range(count)
            ]
            cutoff = random.choice(["2/1", "3/1", "4/1", "6/1"])
            prime = random.choice([2, 3, 5, 7])
            queries = [value for value in ("0/1", "1/1", "2/1", "3/1", "6/1") if Fraction(value) <= Fraction(cutoff)]
            self.check_document(document(points, prime=prime, cutoff=cutoff, queries=queries))


class VerifierMutations(unittest.TestCase):
    """The acceptance checker must reject tampered output.

    These tests keep the checker honest: if it accepted everything, the
    differential tests above would prove nothing.
    """

    def run_check(self, output_document):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "in.json")
            target = os.path.join(directory, "out.json")
            with open(source, "w", encoding="utf-8") as handle:
                json.dump(SQUARE, handle)
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(output_document, handle)
            verify.check(source, target)

    def base_output(self):
        return json.loads(dumps(run(json.dumps(SQUARE))))

    def test_accepts_untouched_output(self):
        self.run_check(self.base_output())

    def test_rejects_wrong_betti(self):
        output = self.base_output()
        output["answers"][0]["betti"] = [1, 0, 0]
        with self.assertRaises(AssertionError):
            self.run_check(output)

    def test_rejects_dropped_bar(self):
        output = self.base_output()
        output["bars"] = [bar for bar in output["bars"] if bar[0] != 1]
        with self.assertRaises(AssertionError):
            self.run_check(output)

    def test_rejects_zero_length_bar(self):
        output = self.base_output()
        output["bars"].append([1, "2/1", "2/1"])
        with self.assertRaises(AssertionError):
            self.run_check(output)

    def test_rejects_unsorted_bars(self):
        output = self.base_output()
        output["bars"] = list(reversed(output["bars"]))
        with self.assertRaises(AssertionError):
            self.run_check(output)

    def test_rejects_unreduced_rational(self):
        output = self.base_output()
        output["bars"][0][2] = "2/2"
        with self.assertRaises(AssertionError):
            self.run_check(output)

    def test_rejects_non_cycle(self):
        output = self.base_output()
        output["answers"][0]["cycles"][0] = output["answers"][0]["cycles"][0][:-1]
        with self.assertRaises(AssertionError):
            self.run_check(output)

    def test_rejects_coefficient_out_of_range(self):
        output = self.base_output()
        output["answers"][0]["cycles"][0][0][1] = 2
        with self.assertRaises(AssertionError):
            self.run_check(output)

    def test_rejects_reversed_endpoints(self):
        output = self.base_output()
        cycle = output["answers"][0]["cycles"][0]
        cycle[0] = [["b", "a"], 1]
        with self.assertRaises(AssertionError):
            self.run_check(output)

    def test_rejects_duplicate_cycle(self):
        output = self.base_output()
        cycles = output["answers"][0]["cycles"]
        cycles.append(cycles[0])
        with self.assertRaises(AssertionError):
            self.run_check(output)


if __name__ == "__main__":
    unittest.main()
