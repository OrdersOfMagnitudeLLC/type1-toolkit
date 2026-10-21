# GST 0-surgery-friend check (Prop 2.3 route)

Date: 2026-10-16. Question: does the GST knot appear in, or match any manifold in,
the existing SPC4 friend-pair data? Datasets:

- `/mnt/HIVE/FrontierMath/SPC4/kegel_spreer_results.json` — 41 Kegel–Spreer
  friend pairs (AKW Table 1 census), 82 knot entries, each with census name,
  PD file, and s-invariants (s0, s2, s3, sqo).
- `/mnt/HIVE/FrontierMath/SPC4/data/low_crossing_friends.csv` — 1867 rows
  `Kname, "PD code"` (crossing counts up to 1259).

GST reference: 48 crossings, complement hyperbolic, **vol 23.8448997956**,
isosig `DLLLLLvvLMMzPPAAMQQPccdgghmrosuqouyqsrxz...`, invariants s2=s3=0,
sqo=(0,0), s0 infeasible locally.

## Results

### 1. GST present in either dataset?

- **kegel_spreer_results.json: NO.** No exact PD match among the 82 entries
  (all pair-member PD files compared as crossing-set equality against the
  GST PD — none with 48 crossings matched).
- **low_crossing_friends.csv: NO.** Only 4 rows have 48 crossings
  (K10a92, K11n32, K11n85, K12n41); none is an exact PD match, and their
  exteriors are different manifolds (vols 17.92, 17.59, 15.59, 14.97 —
  all ≠ 23.84, all isosigs differ). So GST is not disguised as an
  unreduced census-knot diagram either.

### 2. Knots sharing concordance invariants with GST

GST's measurable profile is `{s2: 0, s3: 0, sqo: (0,0)}`. **51 of the 82**
pair members match it exactly — and all 51 also have s0 = 0 (computed on
their smaller diagrams during the earlier census). Interpretation: matching
s-invariants is necessary but far from sufficient for friendship — these
knots merely share "no measured obstruction to being slice", the same
class GST sits in. The 0-surgery criterion (what Prop 2.3 actually needs)
cannot be read off these invariants.

### 3. Complement-manifold match via SnapPy volume

Volumes computed from each pair member's PD-file exterior (census-name
`Manifold()` loading segfaulted — OOM in the compiled layer — so PD
exteriors were used instead):

| statistic | friend-pair members | GST |
|-----------|--------------------|-----|
| vol range | 4.0569 – 9.3557 (77 entries with PD files) | **23.8449** |
| isosig matches | 0 | — |
| vols within 0.01 | 0 | — |
| CSV 48-xing rows | vols 14.97–17.92 | — |

**No manifold in the friend-pair data has GST's complement** — no isosig
match, no volume collision (largest dataset vol 9.36 vs GST's 23.84).
Caveat: complement volume isn't a necessary friend condition (friends
share 0-*surgery*, and the L/R complements in this very dataset have
different vols, e.g. pair 8: 6.91 vs 8.36), so this rules out only
identity-by-isotopy, not friendship per se.

## Verdict

- GST is **not** in either dataset and shares a complement with nothing
  there.
- 51 knots are *invariant-compatible* with GST (s2=s3=sqo=0, s0=0), but
  s-invariants can't certify shared 0-surgery — the real friend test
  (`S^3_0(K1) ≅ S^3_0(K2)`, via annulus twisting or isometry of the
  surgered manifolds) hasn't been run against GST.
- The Prop 2.3 route (find a 2-component R-link `U ∪ GST`) remains open
  but can't be executed from this data: the friend machinery keyed on
  *knot* pairs, whereas the needed object is a *link* whose annulus-twist
  partner is an unknot-containing R-link.
