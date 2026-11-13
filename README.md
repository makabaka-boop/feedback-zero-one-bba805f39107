# Persistent homology backend (0–2 dimensional)

A self-contained Python backend that reads a UTF-8 JSON document describing a
weighted Vietoris–Rips complex and writes reduced persistent homology together
with per-query Betti numbers and an explicit basis of `H_1`.

Everything is exact: coordinates, scales and comparisons use
`fractions.Fraction`, and homology is computed over the prime field `F_p` for
the requested `p ∈ {2, 3, 5, 7}`. Boundary operators, the reduction, and the
change-of-basis tracking are implemented in this repository; no topology
solver or third-party package is used (standard library only, Python ≥ 3.9).

## Running

```bash
python3 persist.py INPUT.json OUTPUT.json      # standalone wrapper
python3 -m persist INPUT.json OUTPUT.json      # package entry point
```

The output file is replaced atomically: on any failure (invalid JSON, invalid
values, I/O error) the previous output file is left byte-for-byte unchanged and
the process exits with status 1, printing a single JSON error object to stderr.
Success exits 0.

Independent acceptance check (shares no computation code with the backend):

```bash
python3 tools/verify.py INPUT.json OUTPUT.json
```

Tests:

```bash
python3 -m unittest discover -s tests
```

## Input format

```json
{
  "points": [{"id": "a", "xyz": ["0/1", "0/1", "0/1"]}, ...],
  "prime": 2,
  "cutoff": "2/1",
  "queries": ["1/1", "2/1"]
}
```

* The document is a JSON object with exactly the members `points`, `prime`,
  `cutoff`, `queries`; all are required and no other members are allowed.
* Each point is an object with exactly `id` and `xyz`. `id` is a non-empty
  ASCII alphanumeric string, unique across points. `xyz` has exactly three
  `p/q` strings (`x`, `y`, `z`).
* `p/q` means an optionally negative integer numerator, `/`, and a positive
  integer denominator, ASCII digits only: `"0/1"`, `"6/4"`, `"-6/4"`, `"12/8"`.
  Values are normalised internally, so `"6/4"` equals `"3/2"`.
* `prime` is a JSON integer (not a boolean) and must be 2, 3, 5 or 7.
* `cutoff` is a non-negative `p/q`; every query is a `p/q` in `[0, cutoff]`.
* `points` and `queries` may be empty. Coincident points are still distinct.
* JSON is parsed strictly: no duplicate object members, no `NaN`/`Infinity`, no
  trailing commas or comments.

## Complex and filtration

* Vertices enter at scale `0`.
* Every subset of 2, 3 or 4 points enters at the **maximum pairwise squared
  Euclidean distance** among its members, and only if that value is `≤ cutoff`.
  Simplices above the cutoff never enter.
* All simplices with the same birth scale are simultaneously effective.
* Interior order within a dimension is `(birth scale, ascending vertex tuple)`,
  which is a valid filtration refinement. Points are ordered by ascending `id`
  (ASCII), and that order fixes every boundary orientation, so the result does
  not depend on the order in which points appear in the input.

Data structures contain only simplices of dimension 0–3 (vertices, edges,
triangles, tetrahedra) — exactly the ones the format defines.

## Persistence

Ordinary (unreduced) homology in dimensions 0, 1 and 2 over `F_p` is computed
with the standard matrix reduction on `d_1`, `d_2`, `d_3`, reduced dimension by
dimension (a boundary matrix is block upper triangular by dimension, so this is
equivalent to reducing the full matrix once). For `d_1` the reduction also
tracks the change-of-basis matrix `V`, so each zero column of the reduced
matrix yields an explicit cycle.

## Output format

```json
{
  "bars": [[0, "0/1", "1/1"], [1, "1/1", "2/1"], [0, "0/1", null]],
  "answers": [{"betti": [1, 1, 0], "cycles": [[[["a", "b"], 1], ...]]}, ...]
}
```

* `bars` items are `[dimension, birth, death]`, interpreting each interval as
  left-closed, right-open `[birth, death)`.
  * Zero-length bars are omitted; duplicates are preserved.
  * A class still alive in the window has `death = null` ("unreduced": it does
    not die inside `[0, cutoff]`).
  * Sorted by dimension, then birth, then death, with `null` deaths last.
  * Scales are reduced `p/q` with positive denominator.
* `answers` keeps the **original query order**. Each item has
  `betti = [b0, b1, b2]` at that query scale and `cycles`, a basis of `H_1`.
* Cycles: a list of `[[start_id, end_id], coefficient]` items.
  * Endpoints are ascending by ASCII id (`start_id < end_id`).
  * Items are sorted lexicographically by the endpoint pair.
  * Zero coefficients are omitted; coefficients are in `1 .. prime-1`.
  * The basis is arbitrary but must be independent in the boundary space and
    span it; `len(cycles) == betti[1]` always holds.

### H1 basis construction

For a query scale `t`, take the classes whose bar satisfies
`birth ≤ t < death`. The reported cycle for each such class is the
corresponding zero column of the reduced `d_1`, expressed in the original
filtration-ordered edges (via the tracked `V`). Their images are linearly
independent modulo `im d_2`, and their number equals `b1`, so they form a basis.

### Reference example

The unit square at `z = 0` with `prime = 2`, `cutoff = "2/1"` and queries
`1` and `2` (see `examples/square.json`):

```json
{
  "bars": [[0,"0/1","1/1"],[0,"0/1","1/1"],[0,"0/1","1/1"],[0,"0/1",null],[1,"1/1","2/1"]],
  "answers": [
    {"betti":[1,1,0],"cycles":[[[["a","b"],1],[["a","d"],1],[["b","c"],1],[["c","d"],1]]]},
    {"betti":[1,0,0],"cycles":[]}
  ]
}
```

`H_1` is `[1/1, 2/1)` as required (the full square cycle is present at scale 1
and filled by the triangles at scale 2), and the Betti numbers at queries 1 and
2 are `[1,1,0]` and `[1,0,0]`.

## Errors

Rejections report the JSON path and, when the failure belongs to an array
element, its zero-based index. Syntax errors are reported at `$`.

```
{"path": "$.points[2].xyz[1]", "index": 1, "message": "xyz[1] must match integer/integer, got 'x'"}
{"path": "$.prime", "message": "prime must be one of 2, 3, 5, 7"}
{"path": "$", "message": "invalid JSON: expected ',' or '}' (at offset 57)"}
```

Within the CLI, the same object is printed to stderr with the explanation also
under the key `error`.

## Layout

| Path | Purpose |
| --- | --- |
| `persist/jsonparse.py` | strict JSON parser with paths and duplicate-member detection |
| `persist/rational.py` | `p/q` grammar, parsing and reduced formatting |
| `persist/schema.py` | complete payload validation |
| `persist/complex.py` | Vietoris–Rips simplex enumeration and filtration order |
| `persist/linalg.py` | sparse `F_p` column reduction, rank and nullspace helpers |
| `persist/persistence.py` | boundary operators, reduction, bars, H1 cycles |
| `persist/pipeline.py` | validation → computation → JSON, atomic file replacement |
| `persist/__main__.py`, `persist.py` | command-line entry points |
| `tools/verify.py` | independent acceptance checker (dense ranks, cycle validation) |
| `tests/test_persist.py` | unit, rejection, atomicity and differential tests |

## Notes and limits

Enumeration is adjacency based, so cost follows the number of simplices under
the cutoff rather than `C(n, 4)`; the reduction is the classical sparse
algorithm and is intended for documents whose complex has up to roughly `10^6`
tetrahedra. Homology is over `F_p` only, as specified.
