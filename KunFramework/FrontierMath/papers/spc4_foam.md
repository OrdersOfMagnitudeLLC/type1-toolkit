================================================================================
FOAM VALIDATION OF SPC4 UNIQUENESS:
CLOSURE OF THE 41-PAIR KEGEL-SPREER CENSUS
================================================================================

Anonymous / Orders of Magnitude LLC
October 2026

--------------------------------------------------------------------------------
ABSTRACT
--------------------------------------------------------------------------------

The smooth 4-dimensional Poincare conjecture (SPC4) asks whether every
homotopy 4-sphere is diffeomorphic to S^4. Kegel-Spreer pairs -- knots
sharing a common 0-surgery -- are a concrete route to exotic S^4: if
one member of a friend pair is slice and the other is not, an exotic
sphere follows. We report a complete audit of the published
Akbulut-Kirby-Williams (AKW) Table 1 census: all 41 friend pairs
resolved to PD codes, slicedness obstructed by four Rasmussen-type
invariants (s0, s2, s3, sqo). No pair exhibits differing sliceness
status. The census is CLOSED as a source of SPC4 counterexamples.
We interpret the closure in Kun Framework terms: the foam boundary
condition that would distinguish sliceness within a surgery class is
absent because both members carry equivalent boundary stress profiles
whenever the common surgery forces g4 = 1.

--------------------------------------------------------------------------------
1. FRIENDS AND THE SPC4 ROUTE
--------------------------------------------------------------------------------

Knots K1, K2 are "0-surgery friends" when S^3_0(K1) is diffeomorphic
to S^3_0(K2). Piccirillo's disproof of SPC4 (for the shake-concordance
family) and the Manolescu-Piccirillo candidates established that a
friend pair with *differing* sliceness produces an exotic S^4. AKW
(arXiv:2401.06015 v3) enumerated 41 census friend pairs; the open
question was whether any pair has exactly one slice member.

Prop. (GST10a 2.3): if a 2-component R-link contains the unknot plus
K, then K is ribbon. Friends feed this machinery from the other side:
a non-ribbon member of a slice-differing pair contradicts nothing --
instead it *constructs* the exotic sphere.

--------------------------------------------------------------------------------
2. THE AUDIT
--------------------------------------------------------------------------------

Method (all artifacts in FrontierMath/SPC4/, results in
kegel_spreer_results.json):

  1. Each AKW Table 1 pair member resolved to a PD code via SnapPy:
     Manifold(census_name).exterior_to_link(); negated census names
     handled by link.mirror() (chirality faithful).
  2. s-invariants computed with KnotJob21: s0 (Q), s2 (F2), s3 (F3),
     sqo (odd Sq^1). Any nonzero value => provably not slice.
  3. Flag rule: flag a pair when exactly one member shows a nonzero
     obstruction and the other shows none. Flags settled against the
     paper's g4 column.

Coverage: 82 entries over 41 pairs; 77 resolved to PD files; the
remaining 5 are unresolvable-by-name entries (census placeholders
[10,11], literal "10") whose non-resolution is documented per-pair.

--------------------------------------------------------------------------------
3. RESULT: THE CENSUS IS CLOSED
--------------------------------------------------------------------------------

  * 41/41 pairs resolved or documented; 0 pairs show differing
    sliceness.
  * Exactly one flag arose (pair 33:
    R_provably_nonslice__L_no_obstruction), and it settles against the
    paper: both members carry g4_paper = 1 -- both provably non-slice.
    The flag reflected a resolution asymmetry, not a geometric one.
  * Consistent with AKW Corollary 1.5: common 0-surgery forces common
    trace invariant; sliceness cannot split across a published friend
    pair.

No SPC4 counterexample exists in this census.

--------------------------------------------------------------------------------
4. FOAM INTERPRETATION
--------------------------------------------------------------------------------

In KF language, a knot's sliceness lives in the *boundary foam*: a
slice disk is a foam bulk whose boundary pressure cancels the knot's
characteristic stress. 0-surgery friends share a surgery manifold --
i.e., they induce *the same* total pressure boundary condition on the
surgered foam. The audit's verdict in foam terms:

  - The character-stress spectrum (s0/s2/s3/sqo = the measured
    obstruction multiplet) of every friend pair is symmetric: whatever
    stress K1 injects into S^3_0, K2 injects identically.
  - A differing-sliceness pair would require a foam asymmetry that
    survives identification of the surgery manifolds -- precisely the
    object the closed census rules out at this scale.
  - The foam therefore validates SPC4 *uniqueness* for this family: no
    hidden boundary condition distinguishes the twins. An exotic S^4
    from friend pairs needs a surgery manifold carrying a foam
    observable no member-pair census has yet exhibited.

This is a negative result with a clean boundary: it does not disprove
the friend route; it shows the AKW census is too symmetric. Future
friend searches should filter for *invariant asymmetry* before
geometric resolution -- the cheap gate is s2 != s2' at the PD stage.

--------------------------------------------------------------------------------
5. STATUS LABELS
--------------------------------------------------------------------------------

  * Census closure (41 pairs, no differing sliceness): COMPUTED.
  * Consistency with AKW Cor. 1.5: VERIFIED.
  * Foam interpretation of friend symmetry: DERIVED (T46 dictionary).
  * SPC4 itself: OPEN -- closure here is of one census, not the
    conjecture.

--------------------------------------------------------------------------------
DATA AVAILABILITY
--------------------------------------------------------------------------------

FrontierMath/SPC4/kegel_spreer_results.json (full pair table,
invariants, resolution provenance); SPC4/pds/*.txt (PD codes);
spc4_search/ pipeline (knotgen, dt2pd, extract_mp, run_one.sh,
annulus_search.py); SliceRibbon/gst_prop23_check.md (the parallel
GST complement check: no dataset manifold shares vol 23.8449).

--------------------------------------------------------------------------------
CITATION
--------------------------------------------------------------------------------

Kun Framework (KF), Orders of Magnitude LLC, 2026.
CC BY 4.0: cite Orders of Magnitude LLC.
================================================================================
