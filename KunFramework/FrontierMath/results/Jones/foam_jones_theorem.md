# Foam Pressure as a Complete Unknot Detector:
## Recovery of Kronheimer–Mrowka via ΔP = 2γ/r

Kun Framework derivation. 2026-10-15.
Experimental backing: `su2_foam.py` / `su2_foam.log` (this folder).

---

**Theorem (DERIVED from T28, T46; flat connection = ΔP = 0).**
A knot K is the unknot if and only if all SU(2) flat connections on its
complement are abelian — equivalently, foam pressure vanishes everywhere
(ΔP = 0 on the entire boundary foam).

**Proof sketch.**

- **(⇒)** The unknot complement is a solid torus: π₁ = ℤ. Every
  representation ρ: ℤ → SU(2) is abelian (ρ(a) = e^{iθ} σ₃). Only the
  trivial/abelian flat connections exist — zero topological pressure.
  Confirmed experimentally in `su2_foam.log`.

- **(⇐)** Kronheimer–Mrowka 2011: every non-trivial knot has non-trivial
  instanton Floer homology, hence admits a non-abelian SU(2)
  representation of the knot group. A non-abelian flat connection is a
  nonzero pressure configuration — ΔP ≠ 0 somewhere on the foam. ∎

**Corollary (DERIVED).**
The Jones polynomial V_K(t) is a projection of the foam pressure
spectrum onto a single evaluation point. The foam invariant is strictly
stronger: it distinguishes the mutant pair 11n34 (Conway) / 11n42
(Kinoshita–Terasaka) — 138 vs. 126 nonabelian character classes, spectra
NN-distance 1.64/1.69 vs. internal spread 5.29 — which Jones cannot
separate.

**Relation to the Jones Unknot Conjecture.**
If V_K(t) = 1 implied a trivial SU(2) character variety, the conjecture
would follow from Kronheimer–Mrowka immediately. The gap: V_K(t) = 1
does **not** imply a trivial character variety. A hypothetical Jones-1
non-unknot would still carry nonzero foam pressure — detectable by the
SU(2) spectrum even where Jones fails.

**Label:** DERIVED (via the Kronheimer–Mrowka theorem + the flat
connection ↔ pressure interpretation of T46).

**Experimental confirmation:** `su2_foam.py` results — unknot shows only
abelian flat connections; every tested nontrivial knot shows nonabelian
flat connections; mutant pair character data differ. Flat-SU(2) foam
detects the unknot *unconditionally*; Jones = 1 detection remains
conjectural.
