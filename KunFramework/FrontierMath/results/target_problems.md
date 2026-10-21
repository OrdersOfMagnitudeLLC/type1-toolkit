# Target FrontierMath Open Problems — Full Statements

Source: Epoch AI FrontierMath Open Problems dataset (CC-BY).

## polya-laplacian-spectrum (full_problem)

Let Omega be a bounded open set in R^2 with Lipschitz boundary, and let

    0 < lambda_1(Omega) <= lambda_2(Omega) <= ...

be the Dirichlet eigenvalues of the positive Laplacian on Omega, counted with
multiplicity. Consider the inequality

    lambda_k(Omega) >= 4*pi*k / |Omega|,

where |Omega| is the area of Omega.

## Task

Find a bounded polygonal domain P in R^2 with rational vertices, together with an
index k >= 1, that violates this inequality:

    |P| * lambda_k(P) < 4*pi*k.

P is an outer simple polygon with zero or more polygonal holes removed from it, so
it may be multiply connected. Supply a strictly positive rational margin epsilon;
the submission is accepted only if the violation holds with room to spare:

    |P| * lambda_k(P) < 4*pi*k - epsilon.

## Submission

Submit the path to a JSON file with this shape:

    {
      "vertices": [[x1, y1], [x2, y2], ..., [xm, ym]],
      "holes": [ [[..], [..], [..]], ... ],
      "k": <positive integer>,
      "epsilon": <positive rational>,
      "refinement": <non-negative integer>
    }

  * "vertices": the outer simple polygon, vertices in cyclic order.
  * "holes" (optional, default none): simple polygons removed from the outer one.
    Each hole must lie strictly inside the outer polygon, and holes must be
    pairwise disjoint.
  * "k", "epsilon": as above.
  * "refinement": mesh refinement level used to bound lambda_k(P); larger values
    give a tighter bound at higher cost.

Each coordinate and epsilon is an exact rational, given as an integer or as a
string that is a fraction "p/q" or a plain decimal such as "0.25". Floating-point
numbers and scientific notation are not accepted.

---

## lonely-runner (full_problem)

# The Lonely Runner Conjecture

Consider n runners on a circular track of length 1, all starting at position 0 at time t=0, each running at a distinct positive integer speed. The conjecture states:

    For any such set of n runners, there exists a time t at which every runner is at distance at least 1/(n+1) from the starting position 0.

Here, distance on the circular track means the shorter arc length, i.e. the distance of runner k from 0 at time t is

    ||t * vk|| = min( {t * vk}, 1 - {t * vk} )

where {x} denotes the fractional part of x.

## Your task

Find a counterexample: a list of n >= 2 distinct positive integers v1, ..., vn such that for every time t, at least one runner is at distance strictly less than 1/(n+1) from position 0. In other words, there is no moment at which the starting position is "lonely".

## Submission format

Write your answer to a file containing the speeds as integers separated by spaces or newlines, for example:

    3 7 11 15

Submit your file path using the submit tool.

---

## prime-factorization (full_problem)

# Factoring semiprimes faster than the best known classical algorithms

You will be given semiprimes $N = p \cdot q$ to factor, and scored on whether you do so
*faster than the best known classical method could* — by a margin large enough that any
positive score is very likely to be an algorithmic breakthrough, not just a fast
implementation or beefier hardware.

## Submission format

You will submit a single Python file. It may use `sage`. When submitting, you may also
constrain the challenges your program will receive by specifying `logp` and/or `logq`:

| parameter | meaning |
|-----------|---------|
| `logp` | $p$ is chosen with $2^{logp} < p < 2^{logp+1}$ |
| `logq` | $q$ is chosen with $2^{logq} < q < 2^{logq+1}$ |

Both must be between 3 and 3000. If `logp` is given but not `logq`, $q$ is chosen from
the same range as $p$. If `logq` is given but not `logp`, then for each challenge a value
$k$ is chosen randomly with $\min(110, logq) \le k \le logq$ and $p$ is chosen with
$2^k < p < 2^{k+1}$. If neither is given, for each challenge a random $k$ between 110 and
3000 is chosen and $p$ and $q$ are drawn independently from $(2^k, 2^{k+1})$.

## Protocol

The verifier repeatedly:

1. picks a challenge semiprime $N = p \cdot q$ subject to your constraints;
2. runs your program in a fresh process, with `N` in decimal as the sole command-line
   argument, using the same Python interpreter that runs the verifier. That is Sage's
   Python, which is not necessarily the `python` on your `PATH` — find it with
   `sage -c 'import sys; print(sys.executable)'` and test your submission with that
   interpreter, or `import sage` may fail. The wall-clock time of the whole invocation
   is measured, including interpreter startup;
3. reads the last non-empty line of your program's stdout as your answer: the two prime
   factors of `N`, space-separated. Earlier lines of stdout are ignored, so printing
   progress information is fine.

Your program is run on one challenge after another until a total wall-clock budget of 30
minutes is exhausted (at most 20 challenges). Each invocation may use whatever remains of
the 30 minutes, so a program that spends the whole budget on one instance gets exactly
one attempt. If you want to reuse expensive precomputation across challenges, cache it in
a file.

## Scoring

For each correct factorization, the verifier estimates the wall-clock time the best known
classical method would have needed for an instance of that shape — ECM (governed by the
smaller factor) for unbalanced $N$, SIQS/GNFS for balanced $N$ — using a calibrated cost
model of the standard $L$-notation complexities. Your score is the speedup
`classical_time / time_spent`, and it is 0 unless it clears a breakthrough cutoff:

```
cutoff = parallel_allowance · calibration_uncertainty · substantial_speedup
       = 16 · 10 · 1000 ≈ 1.6e5
```

The modeled classical time is single-threaded; `parallel_allowance` discounts the cores
your program may use, `calibration_uncertainty` covers hardware/implementation/model
error, and `substantial_speedup` is the genuine algorithmic gain we insist on. So a
positive score means you factored $N$ over 160000x faster than the best known classical
method plausibly could on this machine. Choose instance sizes large enough that this is
possible at all: an instance the model prices at a few seconds cannot clear the cutoff no
matter how fast you factor it.

If there is sufficient time available, your program may be run on multiple challenges.
It will be considered correct if it clears the cutoff on any challenge within the
available time.

## Example submission format

```
import sys

from sage.all import ZZ, factor

N = ZZ(sys.argv[1])
F = factor(N)
# Your answer: the two prime factors of N
print(" ".join(str(p) for p, e in F))
```

---

## slice-ribbon (full_problem)

Your task is to prove that the GST knot is ribbon by exhibiting an explicit ribbon disk.

Starting from the diagram K below, exhibit a sequence of Reidemeister moves and ribbon band moves that ends in a crossingless unlink. If your sequence uses n band moves, the final unlink must have exactly n + 1 components (each ribbon band move splits off one component). Such a sequence is exactly a ribbon-disk certificate for K.

## The starting diagram

K is given by this PD (planar diagram) code — a list of 4-tuples, one per crossing, listing the four strand labels around the crossing counter-clockwise. Your certificate must begin from exactly this diagram:

GST_START_PD = [
  [83, 94, 84, 95], [92, 81, 93, 82], [72, 95, 73, 96], [73, 84, 74, 85], [93, 70, 94, 71],
  [82, 71, 83, 72], [2, 10, 3, 9], [17, 5, 18, 4], [7, 19, 8, 18], [8, 23, 9, 24],
  [3, 25, 4, 24], [31, 1, 32, 96], [27, 12, 28, 13], [38, 5, 39, 6], [6, 37, 7, 38],
  [1, 46, 2, 47], [28, 44, 29, 43], [47, 33, 48, 32], [51, 21, 52, 20], [21, 53, 22, 52],
  [34, 53, 35, 54], [50, 35, 51, 36], [54, 50, 55, 49], [58, 20, 59, 19], [22, 60, 23, 59],
  [33, 60, 34, 61], [57, 36, 58, 37], [61, 49, 62, 48], [55, 63, 56, 62], [63, 57, 64, 56],
  [13, 69, 14, 68], [42, 69, 43, 70], [29, 75, 30, 74], [76, 12, 77, 11], [15, 79, 16, 78],
  [77, 27, 78, 26], [40, 79, 41, 80], [75, 44, 76, 45], [30, 86, 31, 85], [87, 11, 88, 10],
  [16, 90, 17, 89], [88, 26, 89, 25], [39, 90, 40, 91], [86, 45, 87, 46], [65, 80, 66, 81],
  [64, 91, 65, 92], [67, 15, 68, 14], [66, 41, 67, 42],
]

## Certificate format

Write your answer to a file containing a single Python literal — a dictionary:

    {
        "initial": [...],                 # must equal GST_START_PD (lists or tuples both fine)
        "moves": [ <move>, <move>, ... ], # the explicit sequence
        "final_unlink_components": n + 1, # optional; inferred if omitted
    }

Each move is a dictionary carrying its `type`, the data identifying where it is applied, and a `result`: the PD code of the diagram *after* the move. The verifier applies the move to the current diagram itself and checks your `result` agrees up to planar isotopy. A result may be a PD code, `{"pd": [...]}`, or — only for the final crossingless unlink — `{"unlink": n}` (or a plain `[]`, whose component count is then inferred). A PD code cannot record a crossingless (split, unknotted) component, so if a band splits off a trivial circle while other components still have crossings, write that intermediate result as `{"pd": [...], "unlink": k}`, meaning the PD diagram together with `k` extra crossingless unknot components.

Supported moves:

    {"type": "R1", "direction": "remove", "crossing": i, "side": s, "result": [...]}
    {"type": "R1", "direction": "add", "strand": (c, s), "hand": "left"|"right", "result": [...]}
    {"type": "R2", "direction": "remove", "crossing": i, "side": s, "result": [...]}
    {"type": "R2", "direction": "add", "strands": [(c1, s1), (c2, s2)], "result": [...]}
    {"type": "R3", "triple": [(c1, s1), (c2, s2), (c3, s3)], "result": [...]}
    {"type": "band", "band": <spec>, "result": [...]}

Crossing/strand references are (crossing_index, strand_index) with strand_index in {0,1,2,3}. An R3 triple names the three corners of the triangular face the move is applied across; the corners may be listed in any order. A band spec is a Spherogram compressed band string (e.g. "223505_1_1") or a triple accepted by `spherogram.links.bands.core.Band`.

---

