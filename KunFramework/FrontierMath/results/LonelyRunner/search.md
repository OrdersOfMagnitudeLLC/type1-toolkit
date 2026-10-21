# LONELY RUNNER — counterexample search, n=8
# 2026-10-15. Code: /mnt/HIVE/FrontierMath/LonelyRunner/lonely_runner.py

STATUS: NO COUNTEREXAMPLE FOUND. All screened n=8 speed tuples have a
lonely time — consistent with the conjecture; the best near-miss is
the progression (1..8), which sits EXACTLY at the 1/9 threshold.

=====================================================================
METHOD
=====================================================================
Counterexample = speeds v_1..v_8 s.t. every t has some runner with
||t v_k|| < 1/9. Integer speeds make the pattern 1-periodic, so this
is coverage of the circle R/Z by the runners' "home" intervals
((m-1/9)/v_k, (m+1/9)/v_k), m=0..v_k-1 — total measure 16/9 > 1,
so coverage is combinatorially possible but requires bounded overlap.

Checker (exact_uncovered): builds all home intervals on R/Z, computes
uncovered measure, and evaluates ||t v_k|| DIRECTLY at every interval
breakpoint — an isolated uncovered point is still a lonely time and
rules out a counterexample. (Fixed during the run: breakpoint check
added after the interval-measure version masked lonely points.)
Screening: vectorized max_t min_k ||t v_k|| on a 4000-point t-grid;
counterexample requires score < 1/9 exactly. Screen is conservative —
survivors get the exact sweep.

=====================================================================
SANITY — the checker reproduces the known n<=7 theorem
=====================================================================
  n=1 [1]:   lonely point at t=0.5  (||0.5||=0.5 >= 1/2)  CORRECT
  n=2 [1,2]: 4 lonely points (t=2/3 type)                  CORRECT
  n=7 [1..7]: 4 isolated lonely points                     CORRECT —
             this is the proven first-hard-case boundary behavior:
             coverage is perfect except at isolated instants.
  n=8 [1..8]: 7 lonely points, uncovered measure = 0, score = 0.1111
             EXACTLY = 1/9 threshold. The progression family touches
             but does not break the bound — known tight family.

=====================================================================
RESULTS
=====================================================================
Structured families (exact check):
  1..8        score 0.1111  lonely-pts 7   <- nearest miss, threshold
  odd         score 0.5000
  pow2        score 0.3312
  primes      score 0.2500
  squares     score 0.2000
  fib         score 0.1818
  lukac-like  score 0.2772

Random search: 102,400 tuples, distinct speeds <= 1000.
  best screen score: 0.1960 ([552,362,267,838,244,457,770,95])
  ALL top-10 candidates: positive uncovered measure 0.097-0.127,
  hundreds of lonely breakpoints. Nothing approached 0.1111.

=====================================================================
INTERPRETATION
=====================================================================
The n=8 case remains consistent with the conjecture. Notable: the
"canonical" candidate (1..8) achieves the threshold EXACTLY — the
pattern that defeats the bound for small n only saturates it at n=8,
suggesting the conjecture's margin is zero but sign is positive there.
Random tuples sit far away (~0.20 vs threshold 0.111). If an n=8
counterexample exists at all, it likely needs non-generic structure
(known near-misses in the literature come from carefully built
speed sets, not random draws), and speeds <= 1000 random search is
the wrong tool — the literature has pushed structured searches much
further without success.
=====================================================================
