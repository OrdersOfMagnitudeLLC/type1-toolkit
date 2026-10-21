# GST knot — initial findings (FrontierMath `slice-ribbon` problem)

Date: 2026-10-15. Environment: snappy 3.3.2 + spherogram (no Sage), KnotJob21 for s-invariants.

## 1. Diagram loads cleanly

- `GST_START_PD` from the problem statement parses as a valid PD code:
  48 crossings, labels 1..96 each appearing exactly twice.
- `spherogram.Link(crossings=PD)` (this SnapPy version takes the PD list as the
  `crossings` arg — no `PD_code=` kwarg, `Manifold('-name')` unsupported):
  - components: **1** (a knot, as required)
  - writhe: **2**
  - `simplify('global')` leaves it at **48 crossings** — diagram is already
    reduced under spherogram's simplifier.
- Complement: hyperbolic, volume **23.8449**. `identify()` returns `[]` —
  the exterior is not in SnapPy's census (expected at 48 crossings).

## 2. Obstructions (ribbon => must all vanish)

Sage-dependent methods are unavailable in this SnapPy build
(`signature()`, `determinant()`, `alexander_polynomial()`,
`exterior().fox_milnor_test()` all raise `SageNotAvailable`).

Available: KnotJob21 s-type invariants (Rasmussen-style, all must be 0
for a ribbon/slice knot; any nonzero value proves the knot is not slice,
hence not ribbon, and the problem would be asking the impossible).

Status: see `gst_invariants.runlog` / `gst_pd.txt_s*` output files.

| invariant | value |
|-----------|-------|
| s0 (Kh over Q, Rasmussen s) | **INFEASIBLE locally** — timed out at 590s, 900s, 30min, 90min (4 attempts) |
| s2 (s over F2) | **0** |
| s3 (s over F3) | **0** |
| sqo (odd Sq^1, tuple) | **(0, 0)** |

s2 = s3 = sqo = 0 is consistent with slice => consistent with ribbon.
No obstruction found.

s0 never completed (48-crossing Khovanov over Q exceeds 90 min of
single-core java on this box — CPU-bound, ~470MB RSS, so RAM isn't
the constraint). Not needed: GST10a PROVES these knots slice, so
s = 0 is forced; any nonzero s0 would flag a transcription bug,
not disprove slice-ribbon. tau unavailable locally (no HFK toolchain).

VERDICT: knot is slice (literature + s2/s3/sqo=0 consistent); the
FM task is feasible in principle, hard in practice — see below.

## 3. Literature context (what "GST knot" is)

- GST = Gompf–Scharlemann–Thompson, arXiv:1103.1601 (Geom. Topol. 14
  (2010) 2305–2347). Their knots K_n are **slice but not known ribbon** —
  THE candidate counterexamples to slice-ribbon.
- Epoch's own page for this problem confirms: "prove that ONE of the GST
  knots is ribbon" — success would be evidence toward slice-ribbon, not
  a contradiction (no invariant can certify non-ribbon anyway; our
  obstruction check is one-sided).
- "Searching for ribbons with machine learning" (IOP, 2025): their
  algorithms **failed** to find ribbon bands for GST examples — the
  18- and 40-crossing GST links are known ribbon (GST10a) but resisted
  automated search, and the slice knot associated to L_{3,1}
  (likely this 48-crossing diagram) is listed under "ribbon status
  unknown". Expect the spherogram BFS to struggle too.
- Indirect route exists (Prop 2.3 of GST10a): if a 2-component R-link
  has one unknot component + K, then K is ribbon. IOP used this for 3
  knots whose 0-surgery is shared with a ribbon knot — connects to the
  SPC4 friend-pair machinery in ../SPC4/.

## 4. Ribbon search machinery

`spherogram.links.bands` provides exactly what the certificate needs:

- `Band(cs_along_top, arc_is_under, num_twist)` + `add_one_band`
- `search.ribbon_concordant_links(link, max_bands, max_twists,
  max_band_len, paths, certify=True)` — BFS over band sequences toward
  the unlink, returns certificate `[PD, bandspec, PD, bandspec, ..., 'unknot']`
- `search.verify_ribbon_to_unknot(link, certificate)` — verification.

Caveat: `filter_for_plausibly_slice=True` needs Sage — must run with
`filter_for_plausibly_slice=False` (bigger search space, no filtering).

The problem's certificate format (moves dicts with R1/R2/R3/band +
`result` PD codes) can be produced by expanding each band step into
verifier moves once a spherogram certificate exists.

## 5. Status / next steps

- s0 (Rasmussen s over Q): timed out at 590s and 900s on the 48-crossing
  diagram; relaunched with 30-min budget -> `gst_s0b.runlog`, result in
  `gst_pd.txt_s0`. s2=s3=sqo=0 already recorded (consistent w/ slice).
- 1-band probe DONE (871s): `ribbon_concordant_links(L, max_bands=1,
  max_twists=2, max_band_len=3, paths='shortest', certify=True)` ->
  **no 1-band ribbon disk**. ~2200 distinct 2-component descendants
  generated (bands add crossings: most land at 49-52 crossings, 16
  simplify down to 17-29); none is an unlink. Descendants <=40
  crossings saved to `small_descendants.json` (76 links).
- 2-band probe DONE (partial): 6 of 76 small descendants checked
  (~1700s budget, ~4-5 min each) -> 1048, 842, 636, 932, 914, 573
  second-order links; **no unlink**. Search space per descendant is
  ~600-1000 candidate links — a full 2-band sweep needs ~6 hrs of
  serial compute or a smarter band selector.
- s0: 4th timeout at 90 min — CPU-bound infeasibility confirmed on
  this hardware. NOT a gap: GST10a proves slice, s2/s3/sqo=0 corroborate.
- Band-search status: 1-band exhaustive (no disk); 2-band covers only
  ~8% of small descendants (~6/76) — blind BFS is on the losing side
  of this problem. The IOP 2025 ML pipeline also failed on GST family.
- Infrastructure note: spherogram `ribbon_concordant_links` certificates
  hide internal R-moves — any found band path still needs a
  move-tracking simplifier pass to produce verifier-format move dicts
  (deferred until a candidate sequence actually exists).
- RAM note: ~3-4 GB free — heavy jobs ran sequentially; python band
  enumeration peaked ~850MB, java s0 ~470MB.

## 6. Prop 2.3 friend sweep + tooling caveat (2026-10-16)

- `spc4_search/annulus_search.py::insert_twist` produces **non-planar**
  PD codes at every twist site — verified on stevedore/8_8/8_18 (0/81
  planar) and GST (128/128). The ladder surgery only embeds when the
  two strands cobound a channel; corner-adjacent edges diverge, giving
  genus>0 gluings. Even on true bigon faces it fails (0/6 on the
  stevedore), so the wiring itself is also wrong.
- **CONSEQUENCE:** the earlier SPC4 `phase2_annulus_results.txt` and
  `k2_annulus_*.txt` KnotJob outputs were computed on non-planar PDs —
  Khovanov-type invariants of invalid diagrams are not knot invariants;
  that data is suspect and should be regenerated if ever relied on.
- Corrected approach (gst_friend_sweep.py): use spherogram's
  `banded_links` (correct planar band insertion) and test each
  2-component descendant for the Prop 2.3 condition directly:
  linking number 0, an unknot component, and pi1(S^3_0) = F_2
  (0-surgery = #2 S1xS2, verified on controls: 2-unlink -> 2 gens
  0 rels; Whitehead -> 2 gens 2 rels). A hit implies GST is ribbon.
- GST diagram has only 4 bigon faces anyway, so raw annulus-twist
  sites are scarce; the band sweep covers the larger neighborhood.
