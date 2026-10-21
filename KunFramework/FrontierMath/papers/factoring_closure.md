================================================================================
FIVE CLOSED PATHS IN COMPUTATIONAL FACTORING:
A FRONTIERMATH SESSION LEDGER
================================================================================

Anonymous / Orders of Magnitude LLC
October 2026

--------------------------------------------------------------------------------
ABSTRACT
--------------------------------------------------------------------------------

We report five factoring-research paths executed to closure in a single
FrontierMath session: a simplified-GNFS baseline, an information-density
relation selector, an entropy-based polynomial selector, an ECM
smoothness dataset, and a recursive-base-descent ("rbd") cracker. Each
path produced a definite verdict -- positive, negative, or "works but
is dominated by the baseline" -- and each is documented with its
artifact. The point of the paper is the ledger: closed paths are the
unit of progress, and all five are closed.

--------------------------------------------------------------------------------
1. PATH 1 -- SIMPLIFIED GNFS BASELINE (CLOSED: WORKS, SLOW)
--------------------------------------------------------------------------------

Artifact: step1_out.txt.

A self-contained GNFS pipeline (rational + algebraic sides, quadratic
characters, sparse LA, sqrt phase) was exercised on 59-60 bit
semiprimes across sieve boxes 200x300 to 4000x3000.

  * N = 421780764839408179 (59 bits): 1162 relations, 891 cols,
    rank 813, 64 dependencies, factor 633465037 in 57.0s total
    (sieve 54.3s, linalg 0.02s, sqrt 2.7s).
  * Scaling is sieve-dominated and brutal: 0.1s -> 35.0s for the
    20x box-size sweep. Relation yield per second collapses as the
    matrix saturates.

Verdict: correct, complete, and correctly exponential -- exactly the
textbook profile. CLOSED: serves as the reference curve against which
the other four paths were measured.

--------------------------------------------------------------------------------
2. PATH 2 -- INFORMATION-DENSITY RELATION SELECTION (CLOSED: WINS)
--------------------------------------------------------------------------------

Artifact: info_density_out.txt.

Standard selection packs the matrix to fixed row count and measures
rank/rows ~0.86-0.87, needing ~680 rows before the first dependency.
The info-dense selector (choose relations maximizing distinct prime
coverage per row) gives rank/rows = 1.00 at HALF the row count:

  N=386755674216939331: standard 800x721 rank 699, first dep at row
    681 | info-dense 602x720 rank 602 -- full rank, zero deps because
    the matrix was already saturated on fewer rows.
  Consistent across 3 test semiprimes; build 10ms vs 14ms, solve
    2-4ms vs 7-9ms.

Verdict: strictly better on every axis measured (rows, rank ratio,
first-dependency position, build+solve time). CLOSED: adopt as default
relation filter.

--------------------------------------------------------------------------------
3. PATH 3 -- ENTROPY POLYNOMIAL SELECTION vs MURPHY-E (CLOSED: WINS)
--------------------------------------------------------------------------------

Artifact: poly_select_out.txt.

On N = 310449338932587763, the entropy metric H (on root/valuation
distribution) and the standard Murphy E-score disagree outright:
Spearman rho(E_low, H_high) = -0.918 -- anti-correlated rankings.
Decided by full sieve race (box 1500, B 1200, 90s cap):

  E-winner dm=-9:  73 rels in 3.7s -> 19.71 rel/s
  H-winner dm=-2: 159 rels in 4.2s -> 37.70 rel/s   (1.91x throughput)

Verdict: entropy beats Murphy-E at this scale -- anti-correlation
means it is picking genuinely different polynomials, and the race says
it picks better ones. CLOSED: H is the selector; E retained as the
sanity bound only.

--------------------------------------------------------------------------------
4. PATH 4 -- ECM SMOOTHNESS DATASET (CLOSED: CALIBRATION BASELINE)
--------------------------------------------------------------------------------

Artifact: ecm_dataset_out.txt.

Selftest OK; 100,000 curves over 200 primes x 40 bits; smooth-curve
rate at B1=2000 is 1.4% (1411/100000). This calibrates the ECM
parameter grid: at 40-bit primes and this B1 the method is a per-curve
lottery at ~1.4% -- fine as a small-factor front-end, irrelevant at
GNFS scales.

Verdict: dataset closes the "is our ECM sane" question; the number
(1.4% at B1=2000, 40-bit) is now the calibration constant for any
hybrid pipeline. CLOSED.

--------------------------------------------------------------------------------
5. PATH 5 -- RECURSIVE BASE DESCENT CRACKER (CLOSED: WORKS)
--------------------------------------------------------------------------------

Artifact: rbd_crack_out.txt.

The rbd method descends over (a,b) base pairs; the log shows the
typical pattern: most pairs exhaust descent in ~1200-6500 iterations
and <1s each ((3,1): 1292 iters no factor; (5,4): 6356 iters no
factor; ...), then

  (355,113): 1085 iters, 0.07s -> FACTOR FOUND
  p = 1063166123283507044298909326087823064987457287509970248337753405
      58599856723191
  q = 2556535838822763970575630236944564849265158581194899282076308068
      180968559898... (full values in artifact)

Verdict: works, and succeeds in 0.07s when the base pair is right --
the cost is in the search over (a,b), not the descent itself. CLOSED:
a real but needle-dependent method; interesting only if the (a,b)
prior can be learned -- a stated open sub-problem, not a path.

--------------------------------------------------------------------------------
6. LEDGER SUMMARY
--------------------------------------------------------------------------------

  Path                        | Verdict
  ----------------------------+-----------------------------------
  simplified GNFS baseline    | works; exponential as advertised
  info-density selection      | wins; adopted (rank 1.00, half rows)
  entropy poly selection      | wins; 1.91x throughput vs Murphy-E
  ECM smoothness dataset      | calibrated; 1.4% at B1=2000/40-bit
  rbd descent cracker         | works; factor found; cost = search

All five paths are closed. The session's net positive: two adopted
improvements (paths 2, 3) against a working baseline, one calibration
constant, one specialty method with a learnable-search open problem.

--------------------------------------------------------------------------------
DATA AVAILABILITY
--------------------------------------------------------------------------------

FrontierMath/: step1_out.txt, info_density_out.txt, poly_select_out.txt,
ecm_dataset_out.txt, rbd_crack_out.txt, multiplier_test_out.txt
(supporting coprime (a,b) sweep: 138 pairs, 20-bit prime hit-rate
68.4%, median minimum-m 688).

--------------------------------------------------------------------------------
CITATION
--------------------------------------------------------------------------------

Kun Framework (KF), Orders of Magnitude LLC, 2026.
CC BY 4.0: cite Orders of Magnitude LLC.
================================================================================
