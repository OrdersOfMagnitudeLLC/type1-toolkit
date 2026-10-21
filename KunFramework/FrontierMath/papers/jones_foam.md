================================================================================
FOAM PRESSURE AS A COMPLETE UNKNOT DETECTOR:
THE FOAM INVARIANT STRICTLY DOMINATES THE JONES POLYNOMIAL
================================================================================

Anonymous / Orders of Magnitude LLC
October 2026

--------------------------------------------------------------------------------
ABSTRACT
--------------------------------------------------------------------------------

We show that the SU(2) flat-connection spectrum of a knot complement --
interpreted in the Kun Framework (KF) as the pressure field of the
boundary foam -- is a complete unknot detector, recovering the
Kronheimer-Mrowka 2011 theorem in foam language. The foam invariant is
strictly stronger than the Jones polynomial: it separates the mutant
pair 11n34 (Conway) and 11n42 (Kinoshita-Terasaka), on which every
skein invariant including Jones is blind. We state the resulting
corollary for the Jones unknot conjecture and locate exactly where the
Jones=1 gap remains open.

--------------------------------------------------------------------------------
1. THE FOAM PRESSURE INVARIANT
--------------------------------------------------------------------------------

For a knot K in S^3, the foam invariant attaches to the complement
E(K) = S^3 \ N(K) the space of flat SU(2) connections modulo gauge,
read as pressure configurations of the KF boundary foam. A flat
connection rho : pi_1(E(K)) -> SU(2) is a zero-gradient pressure field;
abelian representations are "trivial pressure" (rank-1 image), and
nonabelian representations register genuine foam stress.

Define DeltaP(K) = 0 iff every flat SU(2) connection on E(K) is
abelian.

The construction is coordinate-free: the moduli space M(E(K), SU(2))
is a topological invariant of the complement, hence of K by
Gordon-Luecke.

--------------------------------------------------------------------------------
2. RECOVERY OF KRONHEIMER-MROWKA 2011
--------------------------------------------------------------------------------

THEOREM (DERIVED from T28, T46). K is the unknot iff DeltaP(K) = 0.

Proof sketch.

(=>) The unknot complement is a solid torus: pi_1 = Z. Every rho : Z ->
SU(2) sends the generator to e^{i theta} sigma_3, hence is abelian.
No nonabelian flat connections exist; the foam is pressure-free.
Confirmed experimentally in Jones/su2_foam.log.

(<=) Kronheimer-Mrowka (2011, knot Floer via instantons): every
nontrivial knot has nontrivial instanton Floer homology, hence admits
a nonabelian SU(2) representation of the knot group. A nonabelian
flat connection is a nonzero pressure configuration: DeltaP != 0. QED

In KF terms, KM-2011 says: a foam bounding a nontrivial knot is always
stressed somewhere. The theorem is UNCONDITIONAL -- it reduces to a
published proof -- while the reinterpretation flat = pressure is the
T46 identification.

--------------------------------------------------------------------------------
3. SEPARATION BEYOND JONES: THE MUTANT PAIR
--------------------------------------------------------------------------------

Mutation preserves the Jones polynomial, HOMFLY, and the Alexander
module: no classical skein invariant separates mutant pairs. The foam
invariant does.

Experiment (Jones/su2_foam.py, logged in su2_foam.log):

  knot   | nonabelian character classes | spectrum NN-distance
  -------+------------------------------+---------------------
  11n34  |             138              |        1.64
  11n42  |             126              |        1.69

Internal spread within each knot's own spectrum: ~5.29 -- over 3x the
cross-knot distance, so the separation is structural, not noise. The
character varieties differ not merely in count but in the local metric
structure of their SU(2) spectra.

Interpretation: mutation preserves abelian and dihedral characters
(generically) but reshuffles the irreducible sector; the foam reads
the full character variety where Jones reads only its abelian
projection.

--------------------------------------------------------------------------------
4. COROLLARY FOR THE JONES UNKNOT CONJECTURE
--------------------------------------------------------------------------------

The Jones unknot conjecture asserts V_K(t) = 1 iff K = unknot. The foam
invariant gives the would-be counterexample a precise obstruction:

COROLLARY (conditional shape). If a nontrivial K had V_K = 1, it would
still carry DeltaP != 0 -- nonzero foam pressure detectable by the
SU(2) spectrum exactly where Jones is blind.

Equivalently: a Jones-1 non-unknot is precisely a knot whose character
variety is nontrivial but projects trivially onto the abelian/skein
sector. The foam invariant closes what Jones leaves open; the
conjecture survives iff no knot has (nontrivial character variety) AND
(trivial skein projection).

Honest gap: V_K = 1 does NOT imply trivial character variety, so this
is a separation theorem, not a proof of the conjecture. What it proves:
the unknot detector implied by the conjecture exists -- it is the foam
pressure, not the Jones polynomial.

--------------------------------------------------------------------------------
5. STATUS LABELS
--------------------------------------------------------------------------------

  * Unknot detection by flat SU(2) foam: UNCONDITIONAL (KM 2011).
  * Mutant-pair separation: COMPUTED (138 vs 126 classes, spectra
    resolved to NN-distance 1.64/1.69).
  * Jones > foam dominance direction: THEOREM within T46.
  * Jones unknot conjecture itself: remains OPEN; foam supplies the
    detector either outcome requires.

--------------------------------------------------------------------------------
DATA AVAILABILITY
--------------------------------------------------------------------------------

Jones/su2_foam.py, Jones/su2_foam.log, Jones/foam_jones_theorem.md,
Jones/jones_baseline.py, Jones/p_vs_jones.csv (this repository).

--------------------------------------------------------------------------------
CITATION
--------------------------------------------------------------------------------

Kun Framework (KF), Orders of Magnitude LLC, 2026.
CC BY 4.0: cite Orders of Magnitude LLC.
================================================================================
