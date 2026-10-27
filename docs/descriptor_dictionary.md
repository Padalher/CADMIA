# CADMIA Dictionary — Terms and concepts of computational theoretical physical chemistry

> A concise encyclopedia of the terms that appear in CADMIA and in its reports.
> Written for readers new to quantum chemistry and computational catalysis: each entry answers **what it is**, **what it is for**, **how it is computed** (equations where they help) and **where to read more** (real, accessible sources).
>
> Each entry explains the term, gives the equation and links it to an example from the program itself.
>
> Notation: `ρ` electron density, `E` energy, `N` number of electrons, `r` position.
> Units: 1 hartree (Eh) = 27.2114 eV = 627.5095 kcal/mol.

## Index

1. [Electronic structure and DFT](#1-electronic-structure-and-dft)
2. [Frontier orbitals and gap](#2-frontier-orbitals-and-gap)
3. [Conceptual DFT (CDFT) and reactivity](#3-conceptual-dft-cdft-and-reactivity)
4. [Acids and bases (theory and connection with CDFT)](#4-acids-and-bases-theory-and-connection-with-cdft)
5. [Atomic charges and bond orders](#5-atomic-charges-and-bond-orders)
6. [Fields and electronic properties](#6-fields-and-electronic-properties)
7. [Geometric structure of the cluster](#7-geometric-structure-of-the-cluster)
8. [Vibrational spectroscopy (IR)](#8-vibrational-spectroscopy-ir)
9. [Thermochemistry, bond energies and adsorption](#9-thermochemistry-bond-energies-and-adsorption)
10. [Materials chemistry: metals and oxides](#10-materials-chemistry-metals-and-oxides)
11. [Computational methods and tools](#11-computational-methods-and-tools)
12. [Units, files and formats](#12-units-files-and-formats)
13. [Consolidated bibliography](#13-consolidated-bibliography)

---

## 1. Electronic structure and DFT

### Electron density (ρ)
**What it is.** The probability of finding electrons at each point in space, `ρ(r)`. Its integral over all space gives the total number of electrons: `∫ ρ(r) dr = N`.
**What it is for.** It is the central variable of all DFT: instead of the wavefunction in 3N coordinates, a function of 3 coordinates is enough. From `ρ(r)` one obtains the electrostatic potential, atomic charges, Fukui indices and the deformation density.
**In CADMIA.** Read from cube files (`.cube`) and used for maps (MEP, Fukui, Δρ). See `cadmia/io/cube.py`.

### DFT — Density Functional Theory
**What it is.** A quantum-mechanical method that obtains the energy and properties of a system from its electron density `ρ(r)`, not from the wavefunction.
**Foundation.** The Hohenberg–Kohn theorems (1964) prove that the ground-state energy is a unique *functional* of `ρ(r)`. Kohn–Sham (1965) make it practical by introducing auxiliary orbitals of non-interacting particles.
**What it is for.** It is the workhorse of computational chemistry: a good balance of cost and accuracy for molecules, clusters and solids.
**Sources.** Hohenberg & Kohn, *Phys. Rev.* **136**, B864 (1964); Kohn & Sham, *Phys. Rev.* **140**, A1133 (1965); Parr & Yang, *Density-Functional Theory of Atoms and Molecules*, Oxford (1989).

### Exchange–correlation functional / Hybrid functional
**What it is.** The "unknown" part of the DFT energy, which approximates the exchange and correlation effects between electrons. A **hybrid functional** mixes a fraction of exact exchange (Hartree–Fock type) with DFT exchange.
**Example in CADMIA.** **PBE0** is the default hybrid (25 % exact exchange on top of PBE).
**What it is for.** The choice of functional fixes the accuracy: hybrids such as PBE0 give good orbital energies, gaps and geometries for oxides and transition metals.
**Sources.** Adamo & Barone, *J. Chem. Phys.* **110**, 6158 (1999) — PBE0.

### Basis set
**What it is.** The set of mathematical functions used to represent the orbitals. "More basis" = more flexibility and accuracy, but more cost.
**Example in CADMIA.** **def2-TZVP** (triple-zeta with polarization) for production calculations; `def2/J` is the auxiliary basis used to speed up the integrals (RIJ).
**Sources.** Weigend & Ahlrichs, *Phys. Chem. Chem. Phys.* **7**, 3297 (2005) — def2 basis sets.

### Level of theory
**What it is.** The combination of method + basis set + corrections that fully defines a calculation, e.g. **PBE0/def2-TZVP + D4**. It is the reproducible "recipe".
**What it is for / golden rule.** Any energy comparison (E_ads, ΔG, Fukui) **is valid only if all species were computed at the same level of theory** and at the same temperature.
**In CADMIA.** The presets (`--preset cdft|analysis`) and `cadmia prepare` fix a homogeneous level for all species of a study.

### Molecular orbital (MO) / Kohn–Sham orbital
**What it is.** One-electron wavefunction describing how an electron is distributed in the molecule. Each MO has an energy (eigenvalue) and holds up to 2 electrons.
**What it is for.** The shape and energy of the MOs explain bonding, reactivity and spectra. The frontier MOs (HOMO/LUMO) govern chemical reactivity.
**In CADMIA.** The eigenvalues are parsed from ORCA outputs (`cadmia/io/electronic.py`).

### Energy levels (electronic) / Energy level
**What it is.** The discrete energies allowed for electrons. In an atom or molecule they are discrete levels (the MO eigenvalues); in a solid they merge into continuous **bands**.
**What it is for.** The filling of levels (up to the HOMO) and the gap to the next level determine stability, color, conductivity and reactivity.
**In CADMIA.** The spectrum of occupied/virtual levels is visualized as **DOS/PDOS** (`cadmia electronic dos/pdos`).

### SCF — Self-Consistent Field
**What it is.** The iterative procedure that solves the Kohn–Sham equations: a density is proposed, the potential is computed, new orbitals are obtained, and the cycle repeats until the density "stops changing" (converges).
**What it is for.** It is the engine of any DFT/HF calculation. If the SCF does not converge, there are no reliable results.
**In CADMIA.** For difficult systems (Ti, charged states) `TightSCF` and more iterations (`%scf MaxIter 512 end`) are recommended.

### Open shell / closed shell — Spin multiplicity
**What it is.** A **closed-shell** system has all electrons paired (multiplicity 1). An **open-shell** one has unpaired electrons (multiplicity ≥ 2). The multiplicity is `2S + 1`, with `S` the total spin.
**What it is for.** It defines the correct electronic state. In open shells there are distinct α and β orbitals.
**Rule in CADMIA.** For open shells, `HOMO = max(HOMO_α, HOMO_β)`. The multiplicity can be resolved automatically from electron parity (`--mult auto`) or set by hand (`--mult N`). See `cadmia/core/spin.py`.

---

## 2. Frontier orbitals and gap

### HOMO — Highest Occupied Molecular Orbital
**What it is.** *Highest Occupied Molecular Orbital*. The filled orbital of highest energy; this is where electrons leave most easily.
**What it is for.** It measures donor/nucleophilic character: a molecule with a high HOMO donates electrons readily.
**In CADMIA.** `HOMO_TCE = −9.1212 eV` is the reference value verified against Multiwfn.

### LUMO — Lowest Unoccupied Molecular Orbital
**What it is.** *Lowest Unoccupied Molecular Orbital*. The empty orbital of lowest energy; this is where the electrons the molecule accepts enter.
**What it is for.** It measures acceptor/electrophilic character: a low LUMO accepts electrons readily.

### HOMO–LUMO gap
**What it is.** The energy difference `E_gap = E_LUMO − E_HOMO`.
**What it is for.** An indicator of stability and reactivity: large gap → stable, unreactive species ("hard"); small gap → reactive and polarizable ("soft"). It is related to chemical hardness: with the η = I − A convention CADMIA uses, Koopmans gives `η ≈ E_gap`.
**In CADMIA.** Computed in `cadmia/core/electronic_descriptors.py` together with μ, η and ω.
**Sources.** Frontier-orbital theory: Fukui, *Science* **218**, 747 (1982).

---

## 3. Conceptual DFT (CDFT) and reactivity

### Conceptual DFT (CDFT)
**What it is.** The branch of DFT that defines **reactivity descriptors** as derivatives of the energy with respect to the number of electrons `N` and the external potential `v(r)`. It connects classical chemical concepts (electronegativity, hardness) with computable quantities.
**What it is for.** Predicting *where* and *how* a molecule will react before doing the expensive reaction calculation.
**In CADMIA.** It is the core of the `cadmia/cdft/` module. The global SCF descriptors must match Multiwfn exactly.
**Sources.** Geerlings, De Proft & Langenaeker, *Chem. Rev.* **103**, 1793 (2003) — reference review.

### Fukui indices (f⁺, f⁻, f⁰)
**What they are.** Functions that indicate which atoms/regions of the molecule are most reactive toward different attacks. They measure how the electron density changes when gaining or losing an electron:

- `f⁺(r) = ρ_{N+1}(r) − ρ_N(r)` → reactivity toward **nucleophiles** (sites that accept e⁻).
- `f⁻(r) = ρ_N(r) − ρ_{N−1}(r)` → reactivity toward **electrophiles** (sites that donate e⁻).
- `f⁰(r) = (f⁺ + f⁻)/2` → **radical** attack.

**Condensed version.** On atoms, using charges `q`: `f⁺_A = q_A(N) − q_A(N+1)` and `f⁻_A = q_A(N−1) − q_A(N)` (finite differences).
**What it is for.** Locating the active site: the atom with the largest `f⁺` is the preferred target of a nucleophile, and so on.
**In CADMIA.** Condensed Fukui from charges in `cadmia/cdft/calculator.py`; the maps on a grid (f± from cubes) are part of the densities plan.
**Sources.** Parr & Yang, *J. Am. Chem. Soc.* **106**, 4049 (1984); Fukui, *Science* **218**, 747 (1982).

### Dual function / Dual descriptor (Δf)
**What it is.** `Δf(r) = f⁺(r) − f⁻(r)`. On a single map it distinguishes electrophilic zones (`Δf > 0`) from nucleophilic ones (`Δf < 0`).
**What it is for.** More robust than f± taken separately; it reveals at a glance the dual character of each region.
**Sources.** Morell, Grand & Toro-Labbé, *J. Phys. Chem. A* **109**, 205 (2005).

### Electronic chemical potential (μ)
**What it is.** The tendency of electrons to "escape" from the molecule: `μ = (∂E/∂N)_v ≈ (E_HOMO + E_LUMO)/2`.
**What it is for.** It is the negative of electronegativity. When two species join, electrons flow from the one with higher μ to the one with lower μ.

### Electronegativity (χ, Mulliken)
**What it is.** `χ = −μ ≈ −(E_HOMO + E_LUMO)/2`. The molecule's affinity for electrons.
**What it is for.** Predicting the direction of charge transfer in a bond or adsorption.

### Chemical hardness (η) and softness (S)
**What it is.** The molecule's resistance to changing its number of electrons: `η ≈ E_LUMO − E_HOMO` (≈ gap; some conventions use half of it). Softness is its inverse, `S = 1/η`.
**What it is for.** It is the quantitative bridge to the HSAB principle (see §4): hard species prefer hard ones, soft species prefer soft ones.
**Sources.** Parr & Pearson, *J. Am. Chem. Soc.* **105**, 7512 (1983) — absolute hardness.

### Electrophilicity index (ω)
**What it is.** `ω = μ² / (2η)`. It quantifies the molecule's capacity to accept charge and stabilize itself by doing so.
**What it is for.** Ranking electrophiles; high ω = strong electrophile.
**Sources.** Parr, von Szentpály & Liu, *J. Am. Chem. Soc.* **121**, 1922 (1999).

### Electrodonating (ω⁻) and electroaccepting (ω⁺) powers
**What it is.** They refine ω by separating the capacity to *donate* charge from the capacity to *accept* it, instead of a single number. With `I` (ionization potential) and `A` (electron affinity):
`ω⁻ = (3I + A)² / [16(I − A)]` (donating), `ω⁺ = (I + 3A)² / [16(I − A)]` (accepting). They satisfy the consistency identity `ω⁺ − ω⁻ = μ`.
**What it is for.** Low ω⁻ = good donor (soft base/nucleophile); high ω⁺ = good acceptor (acid/electrophile). They distinguish the two roles that the global ω merges into one, useful for classifying the acid/base site (§4) and the direction of transfer with the probe.
**In CADMIA.** Fields `omega_minus`/`omega_plus` (`electrodonating_accepting_powers` in `core/electronic_descriptors.py`), reported in the global CDFT block.
**Sources.** Gázquez, Cedillo & Vela, *J. Phys. Chem. A* **111**, 1966 (2007).

### Domingo nucleophilicity index (N)
**What it is.** Empirical nucleophilicity scale referred to a common zero: `N = E_HOMO(molecule) − E_HOMO(TCE)`, with tetracyanoethylene (TCE) as the electron-poorest reference electrophile (`HOMO_TCE = −9.1212 eV`, Multiwfn value at B3LYP/6-31G*). N grows with nucleophilic character.
**What it is for.** Unlike the absolute HOMO, N is comparable across molecules and systems; it is the nucleophilic counterpart of ω. Its condensed local version is `N·f⁻`.
**In CADMIA.** Field `nucleophilicity_ev` (`cdft/calculator.py`); `HOMO_TCE_EV` is a constant verified against Multiwfn and must not be altered without data.
**Sources.** Domingo, Chamorro & Pérez, *J. Org. Chem.* **73**, 4615 (2008).

### Finite differences (FD)
**What it is.** Approximation of the CDFT derivatives by computing the neutral species `N` and its ions `N+1` (anion) and `N−1` (cation) at the same geometry, and subtracting.
**What it is for.** This is how CADMIA obtains the global descriptors and the condensed Fukui functions without analytic derivatives.

---
## 4. Acids and bases (theory and connection with CDFT)

> This section connects the classical chemistry of acids/bases with the descriptors that CADMIA already obtains from CDFT. The classification is **already integrated**: `cadmia cdft` labels each site and summarizes the dominant acid/base pair (see "How CADMIA does it" at the end of the section).

### Brønsted–Lowry theory
**What it is.** An **acid** donates a proton (H⁺); a **base** accepts it. It defines conjugate acid/base pairs.
**What it is for.** Reactions in solution, protonation/deprotonation. On surfaces, hydroxyl groups (–OH) act as Brønsted acid sites.
**Sources.** Brønsted, *Recl. Trav. Chim. Pays-Bas* **42**, 718 (1923).

### Lewis theory
**What it is.** A **Lewis acid** accepts a pair of electrons; a **Lewis base** donates it. It is the most general definition and the relevant one in catalysis on oxides.
**What it is for.** On an oxide, an unsaturated metal cation (e.g. Ti⁴⁺ with incomplete coordination) is a **Lewis acid site** that accepts density from a base (CO₂, NH₃, H₂O…). An oxygen with lone pairs is a **Lewis basic site**.
**Connection with CDFT.** The **LUMO** and the **f⁺** index mark where Lewis acidity lies (it accepts e⁻); the **HOMO** and **f⁻** mark basicity (it donates e⁻). The **dual descriptor Δf** separates, at a glance, acidic sites (`Δf>0`) from basic ones (`Δf<0`).
**Sources.** Lewis, *Valence and the Structure of Atoms and Molecules*, 1923.

### HSAB principle (hard and soft acids and bases)
**What it is.** *Hard and Soft Acids and Bases* (Pearson): "hard acids prefer hard bases, soft acids prefer soft bases". Hard = poorly polarizable, dense charge, large gap/hardness; soft = polarizable, small gap/hardness.
**What it is for.** Predicting affinities and selectivity without computing the full reaction.
**Connection with CDFT.** "Hard/soft" is quantified with the **chemical hardness η** (≈ gap): large η = hard, small η = soft. That is why CDFT and HSAB are two sides of the same coin.
**Sources.** Pearson, *J. Am. Chem. Soc.* **85**, 3533 (1963); Parr & Pearson, *J. Am. Chem. Soc.* **105**, 7512 (1983).

### Acidic site / basic site (on surfaces and clusters)
**What it is.** The specific atom of the cluster that acts as an acid (accepting metal cation) or a base (donating oxygen). Oxides are often **bifunctional** (they have both), which enables acid–base catalysis.
**How it is located in CADMIA.** Cross-referencing the electronic map (LUMO/f⁺ for acidity, HOMO/f⁻ for basicity, Δf to discriminate) with the structural coordination (a metal with **low CN/ECN** is usually a strong Lewis acid). The charge transfer upon adsorption confirms the character.
**Sources (acidity of solids).** Tanabe, *Solid Acids and Bases*, Academic Press, 1970.

### How CADMIA does it

The acid/base reading does not introduce new SCF data: it is a chemical interpretation of descriptors that the CDFT workflow already produces (`cadmia/cdft/acid_base.py`).

| Element | Criterion |
|---|---|
| Label per site | From the **dual Δf**: `Δf > +0.02` → `lewis_acid`; `Δf < −0.02` → `lewis_base`; between both → `amphoteric`; no Δf → `unknown`. The 0.02 threshold is the same one the CDFT workflow uses to consider a site reactive. |
| Strongest acid site | The one with the **largest f⁺** among those labeled as acids (electron-accepting power). |
| Strongest basic site | The one with the **largest f⁻** among the basic ones (donating power). |
| HSAB context | The **global hardness η** accompanies the summary. HSAB is comparative: η does not classify by itself; it is contrasted with the hardness of the probe/reagent (hard with hard, soft with soft). |
| Coordinative unsaturation | The ECN of the acid site is compared against the **mean for the same element in that cluster** (a relative criterion, not an absolute cutoff): below the mean (margin 0.1 ECN) it is reported as coordinatively unsaturated → stronger Lewis acid. |

It appears in the site table of `cadmia cdft` (column `acid/base`), in the "Lewis sites" line of the summary and in the JSON (`acid_base` per site, `acid_base_summary` global).

**Deliberate limitation.** The site ranking is sorted by **|Δf|**, without filtering by character: it is a screening that surfaces the most reactive sites in both directions, and the subsequent DFT decides. Anyone who wants to match the site with the Lewis role of the probe has the per-site label to do so.

---

## 5. Atomic charges and bond orders

> Atomic charges are **not observable**: they depend on the partitioning scheme. That is why CADMIA reports the scheme used and requires the same one in adduct and cluster when comparing.

### Mulliken charges
**What it is.** Distributes electrons according to the population in the basis orbitals. Fast but very sensitive to the basis set.
**Use.** Quick qualitative estimate.

### Hirshfeld charges (stockholder)
**What it is.** Distributes the density at each point among the atoms in proportion to the free-atom density ("stockholder method"). Stable and weakly dependent on the basis set.
**Use.** Reliable charges and charge transfer; a preferred scheme in many CDFT analyses.
**Sources.** Hirshfeld, *Theor. Chim. Acta* **44**, 129 (1977).

### Löwdin charges
**What it is.** A variant of Mulliken with an orthogonalized basis; it reduces some pathologies.

### CHELPG charges
**What it is.** *CHarges from ELectrostatic Potentials, Grid based*. Fits point charges that reproduce the molecular electrostatic potential on a grid.
**Use.** The best ones for electrostatics and force fields.

### NPA / NBO (natural population analysis charge)
**What it is.** Natural population analysis over natural bond orbitals (NBO). Very robust with respect to the basis set.
**In CADMIA.** Not yet available; planned through JANPA/NBO (requires Java).

### Mayer bond indices (orders)
**What it is.** A number that quantifies how many electron pairs two atoms share: ≈1 single bond, ≈2 double, ≈3 triple. Fractional values indicate partial bonds.
**What it is for.** Characterizing bond strength at the adsorption site (how strongly the probe binds to the cluster).
**In CADMIA.** `cadmia bonds` and the section "Mayer bond orders at the binding site" of the report; parsing in `parse_mayer`.
**Sources.** Mayer, *Chem. Phys. Lett.* **97**, 270 (1983).

---

## 6. Fields and electronic properties

### MEP — Molecular electrostatic potential
**What it is.** The potential `V(r)` that a positive test charge would feel at each point, generated by nuclei and electrons. It is usually mapped onto a density surface.
**What it is for.** Seeing at a glance electron-rich zones (red, attract electrophiles) and electron-poor ones (blue, attract nucleophiles); predicting attack sites and hydrogen bonds.
**In CADMIA.** `cadmia electronic mep` and the driver `cadmia density --kind mep` (where ORCA is available).

### Electrostatic interactions / attractions
**What it is.** The force between charged regions: opposite charges attract, like charges repel (`E ∝ q₁q₂/r`). It includes ion–ion, ion–dipole, dipole–dipole attractions and hydrogen bonds.
**What it is for.** They are the primary driver of probe↔cluster approach (before a chemical bond forms) and dominate physisorption. The **MEP** and the **dipole moment** are their direct maps/measures; **CHELPG** charges are fitted precisely to reproduce this electrostatics.

### DOS — Density of states
**What it is.** How many electronic states exist per energy interval. It is obtained by broadening the discrete eigenvalues with Gaussians/Lorentzians.
**What it is for.** Seeing the "band structure" of a cluster: where states accumulate, the position of the gap.
**In CADMIA.** `cadmia electronic dos`; `electronic/dos.py` (Gaussian broadening).

### PDOS — Projected density of states
**What it is.** The DOS decomposed by the contribution of each element or orbital.
**What it is for.** Knowing *which atoms* contribute to the HOMO/LUMO and to bonding; key to understanding which part of the cluster governs reactivity.
**ORCA requirement.** Requires `Print[P_OrbPopMO_L] 1`. CLI `cadmia electronic pdos`.

### Dipole moment
**What it is.** A vector that measures charge separation; its magnitude is given in Debye (D).
**What it is for.** Molecular polarity, electrostatic interactions, change in polarity upon adsorption (`Δ|μ|`).
**In CADMIA.** ORCA parsing in `cadmia/io/electronic.py` (a.u. → Debye, factor 2.5417464519); it appears in the report with its delta.

### Deformation density (Δρ)
**What it is.** `Δρ = ρ_adduct − ρ_cluster − ρ_probe`. It shows how electrons are redistributed when the adsorption bond forms (accumulation vs. depletion).
**What it is for.** Visualizing bond formation: where charge accumulates (the bond) and where it comes from.
**Critical caveat.** The cluster and the probe must be computed as *single points at the frozen geometry of the adduct*, not at their relaxed geometries; otherwise Δρ is contaminated by the geometric change.
**In CADMIA.** `cadmia density --adduct ad.gbw --cluster cl.gbw --probe pr.gbw` (where ORCA is available).

---

## 7. Geometric structure of the cluster

### Cluster / Nanocluster
**What it is.** An aggregate of a few to a few hundred atoms, between the molecule and the bulk solid. Its properties depend strongly on size.
**What it is for.** A catalyst model: an oxide nanocluster (e.g. Al₂TiO₅) represents the active surface without the cost of a periodic calculation.
**In CADMIA.** It is the central object of analysis (`Structure` in `cadmia/core/`).

### Coordination number (CN)
**What it is.** How many bonded neighbors an atom has (an integer count, according to a distance cutoff or covalent radii).
**What it is for.** Characterizing the local environment; poorly coordinated atoms (low CN) tend to be more reactive and, in oxides, are strong Lewis acid sites.
**Sources.** Covalent radii: Cordero et al., *Dalton Trans.*, 2832 (2008).

### ECN — Effective coordination number (Hoppe)
**What it is.** The *continuous* and weighted version of the CN: close neighbors weigh ≈1 and distant ones decay smoothly, according to Hoppe's formula based on distances relative to the shortest one.
**What it is for.** It distinguishes environments that the integer CN confuses; useful for coordination trends vs. cluster size.
**In CADMIA.** `cadmia/structure/structural/` (Hoppe ECN, CN, bonds). Presentation rounding with `Decimal`/`ROUND_HALF_UP`; the raw value is kept. Plots `ecn-trend`/`ecn-heatmap`.
**Sources.** Hoppe, *Z. Kristallogr.* **150**, 23 (1979).

### Bond / Bond length
**What it is.** The distance between two bonded atoms (in Å). It is detected by comparing the distance with the sum of covalent radii plus a tolerance.
**What it is for.** Bond histograms, detection of activation (a bond that lengthens upon adsorption, e.g. the C–O bond of CO₂ that elongates when activated).

### Bond angle
**What it is.** The angle (in degrees) formed by three bonded atoms A–B–C, with the vertex at the central atom.
**What it is for.** Characterizing local geometry (linear 180°, tetrahedral 109.5°, etc.) and its distortion; the **bending** of the probe upon adsorption is a marker of activation (e.g. CO₂ goes from linear 180° to bending when it is activated and takes up charge).

### Coordination polyhedron
**What it is.** The geometric figure formed by the neighbors around a central atom (tetrahedron, octahedron…).
**What it is for.** Classifying crystal-chemical sites and their distortion.
**In CADMIA.** `cadmia/structure/structural/polyhedra.py`. The **type** is decided by geometry, not by the coordination number: Yang's τ₄ for CN=4 (square planar / seesaw / tetrahedron) and Addison's τ₅ for CN=5 (square pyramid / trigonal bipyramid).

### Polyhedral distortion indices (Baur, Robinson σ², ⟨λ⟩)
**What they are.** Three numbers that measure different things, and they should not be confused:

| Index | Equation | Measures | Ideal |
|---|---|---|---|
| **Bond-length distortion** (Baur) | D = (1/n)·Σ \|lᵢ − l̄\| / l̄ | how much the lengths **spread** about their own mean | 0 |
| **Angular variance** (Robinson) | σ² = Σ(θᵢ − θ_ideal)² / (m−1) | **angular** distortion relative to the ideal *cis* angle of the type, in degrees² | 0 |
| **Quadratic elongation** (Robinson) | ⟨λ⟩ = (1/n)·Σ (lᵢ/l₀)² | deformation of the **shape**, relative to the ideal polyhedron of equal volume | 1 |

**The distinction that matters.** Baur normalizes by the **observed mean** length; ⟨λ⟩ normalizes by l₀, the length of the **regular polyhedron of the same volume**. That is why they are not redundant: an octahedron whose six bonds measure exactly the same but whose angles are twisted gives Baur = 0 —the lengths do not spread— and ⟨λ⟩ > 1, because the volume shrank. If ⟨λ⟩ were computed with the mean instead of l₀, it would give exactly 1 and would say nothing that Baur did not already say.

**Missing values.** σ² is NaN where there is no single ideal *cis* angle (trigonal bipyramid, which mixes 90° and 120°); ⟨λ⟩ and the **volume** are NaN for planar polyhedra (triangle, square plane), and ⟨λ⟩ is also NaN where no regular counterpart exists (`seesaw`). Never 0 or 1: a missing value is not disguised as the ideal value. Note that ⟨λ⟩ **is** defined for the trigonal bipyramid, where σ² is not —there is a unique ideal volume even though there is no unique ideal angle— so there it is the only available measure of shape distortion.

**Source.** Robinson, K.; Gibbs, G. V.; Ribbe, P. H. *Science* **1971**, *172*, 567–570 (DOI: 10.1126/science.172.3983.567) — σ² and ⟨λ⟩. Baur, W. H. *Acta Crystallogr. B* **1974**, *30*, 1195–1215 (DOI: 10.1107/S0567740874004560). Yang, L.; Powell, D. R.; Houser, R. P. *Dalton Trans.* **2007**, 955–964 (τ₄). Addison, A. W. et al. *J. Chem. Soc., Dalton Trans.* **1984**, 1349 (τ₅).

---

## 8. Vibrational spectroscopy (IR)

### IR — Infrared spectroscopy / IR spectrum
**What it is.** A technique that measures the absorption of infrared light by the vibrational modes of the molecule. Computationally, the **IR spectrum** is built by placing a peak at each frequency with its intensity and broadening them into a curve.
**What it is for.** The molecule's "fingerprint"; identifying functional groups, following the activation of the probe, and confirming that a geometry is a true minimum.
**In CADMIA.** `cadmia electronic ir` with Lorentzian/Gaussian broadening and a scale factor; parsing in `cadmia/io/vibrational.py`. (Raman, its complement, is on the roadmap.)

### Normal mode of vibration
**What it is.** A collective pattern in which all atoms oscillate at the same frequency and in phase. A non-linear molecule of N atoms has 3N−6 modes (3N−5 if linear).
**What it is for.** Each mode is a potential peak of the IR/Raman spectrum.

### Wavenumber (cm⁻¹) and wavelength
**What it is.** The **wavenumber** `ṽ = 1/λ` (in cm⁻¹) is the standard unit of vibrational spectra: proportional to energy. The **wavelength** `λ` is the distance between crests of the light wave.
**Relation.** `E = h·c·ṽ = h·c/λ`. Typical IR: ~400–4000 cm⁻¹.

### IR intensity
**What it is.** How much light a mode absorbs (km/mol); it depends on the change of the dipole moment during the vibration. Modes with no dipole change are "IR-inactive".

### ZPE — Zero-point energy
**What it is.** The vibrational energy the molecule retains even at 0 K (oscillators never stop): `ZPE = ½ Σ h·ν_i`.
**What it is for.** A correction needed for realistic energies and thermochemistry.
**Important in CADMIA.** It is not a separate step: the ORCA `Freq` calculation already delivers ZPE, H, S and G with the correction included. It is enough to match the level of theory and temperature between species.

### Imaginary frequency
**What it is.** A "negative" frequency (reported as imaginary) indicating that the geometry is not a minimum: one imaginary frequency = transition state; several = higher-order saddle point.
**What it is for / rule.** For adsorption thermochemistry to be valid, all species must have **0 imaginary frequencies** (true minima).
**In CADMIA.** `analysis_agent` warns if the adduct shows imaginary modes.

---
## 9. Thermochemistry, bond energies and adsorption

### Enthalpy (H), Gibbs free energy (G), Entropy (S)
**What it is.** Thermodynamic functions at a given temperature/pressure (298.15 K by default). `G = H − T·S`. ORCA computes them by adding to the electronic energy the translational, rotational, vibrational (including ZPE) and entropic corrections.
**What it is for.** Predicting spontaneity (`ΔG < 0`) and heat of reaction (`ΔH`).
**In CADMIA.** The parser reads `Final Gibbs free energy` → `gibbs_free_energy_hartree` and `Total Enthalpy` → `enthalpy_hartree`.

### Bond energy / interaction energy (what CADMIA computes)
**What it is.** The term "bond energy" has several meanings; CADMIA covers those relevant to adsorption:

- **E_ads (probe adsorption/binding energy)** = `E(adduct) − E(cluster) − E(probe)`. It is the energy with which the probe is bound to the cluster. Negative and large = strong binding.
- **E_ads(BSSE)** = E_ads corrected for basis set superposition (counterpoise).
- **E_def (deformation energy)** of each fragment = cost of distorting the cluster and the probe from their relaxed geometry to the one they adopt in the adduct.
- **E_int (interaction energy)** = `E_ads − E_def(cluster) − E_def(probe)`: the "pure" interaction at frozen geometry. It is a simple **energy decomposition** (E_ads = E_int + ΣE_def).

**What it is NOT (yet).** CADMIA does **not** compute a generic BDE (homolytic dissociation energy of an arbitrary A–B bond). The **Mayer bond order** (§5) gives the bond strength qualitatively.
**In CADMIA.** Everything in `cadmia/agents/analysis_agent.py`; CLI `cadmia analyze`; table in `cadmia/display.py`.
**Sources (reference BDEs).** Luo, *Comprehensive Handbook of Chemical Bond Energies*, CRC, 2007.

### Adsorption energy / enthalpy / free energy (ΔH_ads, ΔG_ads)
**What it is.** The energy change when a probe is attached to the cluster, already including thermal correction:

- `ΔH_ads = H(adduct) − H(cluster) − H(probe)`.
- `ΔG_ads = G(adduct) − G(cluster) − G(probe)`.
- `−TΔS = ΔG_ads − ΔH_ads` (entropic penalty of fixing the probe).

Each `G` is corrected to the rotational symmetry number σ of the geometry itself when the engine used another one: `G ← G + RT·ln(σ_geometry/σ_engine)`. `ΔG_ads(quasi-RRHO)` repeats the difference with Grimme's quasi-RRHO entropy (see below). None of them is computed from a geometry that did not converge.

**What it is for.** Knowing whether adsorption is favorable (`ΔG < 0`) and by how much.
**Validated in CADMIA** (Al₂TiO₅ + CO₂): the four terms are computed with these same formulas.

### Indicative scale of binding/adsorption energies
**What it is for.** Placing a value of `|ΔH_ads|` in its physical regime (approximate values, they vary with the system):

| Regime | `|ΔH_ads|` approx. | Nature |
|---|---|---|
| van der Waals / physisorption | < ~10 kcal/mol (< ~0.4 eV) | no chemical bond; reversible |
| hydrogen bond | ~3–15 kcal/mol | electrostatic/partly covalent |
| weak chemisorption | ~15–30 kcal/mol | incipient chemical bond |
| strong chemisorption | > ~30 kcal/mol (> ~1.3 eV) | chemical bond, activation of the probe |
| typical covalent bond | ~80–110 kcal/mol | reference (BDE of single bonds) |

*Conversion references:* 1 eV ≈ 23.06 kcal/mol; 1 kcal/mol ≈ 0.0434 eV.

### Quasi-RRHO (quasi-rigid-rotor harmonic oscillator)
**What it is.** A correction to the standard harmonic model that treats very low-frequency modes better (which the harmonic approximation overestimates in the entropy).
**What it is for.** More reliable entropies and free energies, especially for flexible complexes.
**Rule.** The same treatment (Quasi-RRHO) and the same T for all compared species.
**Sources.** Grimme, *Chem. Eur. J.* **18**, 9955 (2012).

### Adsorption — Physisorption / Chemisorption
**What it is.** Attachment of a molecule (the **probe**) to a surface/cluster. **Physisorption**: weak, through van der Waals forces (small `|ΔH|`). **Chemisorption**: strong, forms chemical bonds (large `|ΔH|`; the geometry and bonds of the probe change).
**What it is for.** The basis of heterogeneous catalysis: the first step is to adsorb the reactant.

### Active site
**What it is.** The atom or region of the cluster where adsorption/reaction preferentially occurs.
**How it is located.** With Fukui/dual (electronic) and CN/ECN (structural); CADMIA reports the index and symbol of the site atom and its `ΔCN/ΔECN`.

### Probe and Adduct / Complex
**Probe.** The small molecule that is adsorbed to interrogate reactivity (CO₂, CO, NH₃, H₂O…).
**Adduct / complex.** The combined cluster+probe system, already optimized.

### Charge transfer (observed, from populations)
**What it is.** How many net electrons pass from the cluster to the probe (or vice versa) upon adsorption, measured by summing the atomic charges of the probe block in the already computed adduct.
**What it is for.** Diagnosing activation: e.g. the cluster donates density to CO₂ and "activates" it.
**In CADMIA.** Fields `q_transfer_to_probe` and `delta_q_site` (requires the same charge scheme).

### Charge transfer (predicted, Parr–Pearson ΔN)
**What it is.** *Different from the previous one*: it is not measured from the adduct but **predicted** from the global descriptors of the two isolated species, with the two-parabola model. The number of electrons flowing from A to B and the associated stabilization are `ΔN = (μ_A − μ_B) / (η_A + η_B)` (ΔN > 0 ⇒ A donates) and `ΔE = −(μ_A − μ_B)² / [2(η_A + η_B)]`. **Convention.** Here η is CADMIA's `η = I − A`, with no ½, so it is twice the hardness of Parr and Pearson, `η = (I − A)/2 = ½ ∂²E/∂N²`. The original formulas, written with the half hardness, carry an extra 2 in the denominators, `ΔN = (μ_A − μ_B) / [2(η_A + η_B)]` and `ΔE = −(μ_A − μ_B)² / [4(η_A + η_B)]`; the two forms give the same ΔN and ΔE, but do not mix η from one convention with the formula of the other.
**What it is for.** Anticipating the direction and magnitude of cluster↔probe transfer *before* optimizing the adduct, using only μ and η of each fragment. Comparing it with the transfer observed from populations closes the diagnosis.
**In CADMIA.** `compute_charge_transfer` (`core/charge_transfer.py`); fields `delta_n`/`e_stab`.
**Sources.** Parr & Pearson, *J. Am. Chem. Soc.* **105**, 7512 (1983).

### BSSE — Basis set superposition error
**What it is.** An artificial stabilization of the adduct. A finite basis set describes each fragment imperfectly; in the adduct, the cluster can also use the basis functions centred on the probe's atoms (and the probe those of the cluster), so each fragment is described *better* in the adduct than alone. That improvement is not binding, but it lowers `E(adduct)` and makes `E_ads` too negative. It shrinks as the basis grows and vanishes in the complete-basis limit.

**Counterpoise correction (Boys–Bernardi).** Each fragment is recomputed at its geometry **in the adduct**, once in its own basis and once in the full basis of the adduct, with the partner's atoms present only as *ghost atoms* (basis functions without nuclei or electrons):

- `BSSE = [E_cl(own basis) − E_cl(full basis)] + [E_pr(own basis) − E_pr(full basis)]`, all at the adduct geometry. Each bracket is ≥ 0: the extra functions can only lower a variational energy.
- `E_ads(BSSE) = E_ads + BSSE`, so the corrected adsorption energy is less negative.

The correction takes four extra single points per adduct (two per fragment). It does not touch `E_def` or the thermal terms: they are computed for each species in its own basis already.

**What it is for / rule.** It is larger for small basis sets (double-ζ, e.g. def2-SVP) than for large ones (triple-ζ, e.g. def2-TZVP), and it matters most for weak binding, where it can be a large fraction of `E_ads` and even change the ranking of configurations. Its size depends on the system and the basis: compute it, do not assume it. Report the basis and whether `E_ads` is corrected; compare corrected with corrected.

**In CADMIA.** Gaussian: an adduct computed with `Counterpoise=2` is recognized by itself, and its energy is already the corrected one. ORCA has no one-job counterpoise: `cadmia prepare bsse ADDUCT.out --cluster C.out --probe-free P.out` writes the four single points (`<adduct>_cp_{cluster,probe}_{own,full}`, ghost atoms written as `O:`), and `cadmia analyze --bsse` reads their outputs and reports `E_ads(BSSE)`.
**Sources.** Boys & Bernardi, *Mol. Phys.* **19**, 553 (1970); van Duijneveldt, van Duijneveldt-van de Rijdt & van Lenthe, *Chem. Rev.* **94**, 1873 (1994).

---

## 10. Materials chemistry: metals and oxides

### Catalyst and heterogeneous catalysis
**What it is.** A **catalyst** accelerates a reaction without being consumed, offering a path of lower activation energy. In **heterogeneous catalysis** the catalyst (solid) and the reactants (gas/liquid) are in different phases; the reaction occurs at the **surface**.
**What it is for.** It underlies much of the chemical industry (refining, fertilizers, CO₂ conversion…). The elementary cycle is: adsorption → surface reaction → desorption.
**In CADMIA.** The whole workflow (CDFT → adsorption → thermochemistry) serves to evaluate the cluster as a catalyst model.
**Sources.** Chorkendorff & Niemantsverdriet, *Concepts of Modern Catalysis and Kinetics*, Wiley.

### Metals — metallic bonding and d band
**What it is.** In a metal the valence electrons are delocalized in bands. In transition metals the (partially filled) **d band** governs surface reactivity.
**What it is for.** The **d-band model** (Hammer–Nørskov) explains adsorption strength: the higher the energy of the d-band center relative to the Fermi level, the stronger the adsorption.
**Sources.** Hammer & Nørskov, *Adv. Catal.* **45**, 71 (2000); Nørskov et al., *Nat. Chem.* **1**, 37 (2009).

### Metal oxides
**What it is.** Compounds of metal + oxygen with ionic-covalent bonding. They are usually insulators or semiconductors with a defined **gap**. Their surface has cations (Lewis acids) and oxygen anions (Lewis bases).
**What it is for.** Widely used supports and catalysts (TiO₂, Al₂O₃…); their acid-base reactivity and their oxygen vacancies are key.
**Sources.** Henrich & Cox, *The Surface Science of Metal Oxides*, Cambridge, 1994.

### Mixed oxides
**What it is.** Oxides with **two or more** different metal cations in the same lattice (e.g. Al₂TiO₅, aluminum titanate / tialite). The combination creates new coordination environments and acid-base pairs that do not exist in simple oxides.
**What it is for.** Tuning acidity/basicity and stability: the synergy between cations (e.g. strong Ti⁴⁺ acid + basic O) can activate stable molecules such as CO₂, as in mixed Al–Ti oxide clusters.

### Oxidation state
**What it is.** The formal charge assigned to an atom assuming fully ionic bonds (e.g. Ti⁴⁺, O²⁻). It is not the real charge (that is given by Hirshfeld/Mulliken), but it orders redox chemistry.
**What it is for.** Reasoning about redox, vacancies and the metal's ability to accept/donate electrons.

### Oxygen vacancy / coordinative unsaturation
**What it is.** A site where a lattice oxygen is missing, or a cation with fewer neighbors than "normal" (low CN/ECN). It leaves available electrons and orbitals.
**What it is for.** These are the archetypal **active sites** on oxides: maximum Lewis acidity and larger f⁺. CADMIA detects them by cross-referencing low coordination (ECN) with high, localized f⁺/LUMO.

---

## 11. Computational methods and tools

### ORCA
**What it is.** An electronic-structure program (DFT, HF, post-HF) that is free for academic use. CADMIA reads ORCA 6 and Gaussian 16 outputs.
**What it is for.** It performs the optimizations, frequencies, single points and populations that CADMIA then parses and analyzes.
**Sources.** Neese, *WIREs Comput. Mol. Sci.* **2**, 73 (2012); **12**, e1606 (2022).

### Geometry optimization
**What it is.** A search for the minimum-energy geometry by moving the atoms along the gradient.
**What it is for.** Every reliable property starts from an optimized geometry (a minimum).

### Single point
**What it is.** An energy/property calculation at a **fixed** geometry, without moving atoms.
**What it is for.** Essential for the deformation density (frozen geometry of the adduct) and for descriptors at a given geometry.

### xTB / GFN2-xTB
**What it is.** A very fast semiempirical tight-binding method for approximate geometries and energies.
**What it is for.** Pre-screening: exploring many adsorption configurations cheaply before spending DFT on the promising ones.
**In CADMIA.** `cadmia prescreen` (GFN2-xTB, deduplication + energy window).
**Sources.** Bannwarth, Ehlert & Grimme, *J. Chem. Theory Comput.* **15**, 1652 (2019).

### D4 dispersion correction
**What it is.** A term that adds the London/van der Waals forces that standard DFT functionals do not capture. D4 makes the C₆ coefficients **dependent on the atomic charge**.
**What it is for.** Correct geometries and weak interactions. Suitable for Ti and for variable charge states; at fixed geometry it does not alter HOMO/LUMO/Fukui.
**In CADMIA.** Dispersion **default = D4** (configurable).
**Sources.** Caldeweyher et al., *J. Chem. Phys.* **150**, 154122 (2019).

### RIJCOSX
**What it is.** An ORCA approximation that accelerates the calculation of the Coulomb integrals (RIJ, with the auxiliary basis `def2/J`) and of the exchange integrals (COSX, semi-numerical) with negligible loss of precision.
**What it is for.** Making hybrids feasible on large clusters.

### Integration grid (DefGrid)
**What it is.** The numerical mesh on which the DFT functional is integrated. `DefGrid3` is a fine grid.
**What it is for.** Fine grids give stable energies and frequencies (important with transition metals).

### HPC job schedulers: SLURM, PBS/Torque, SGE
**What it is.** Software that manages the resources of a computing cluster (HPC): it receives jobs, places them in a **queue**, assigns them to free nodes according to priority and runs them. They are submitted with a batch *script* (`#SBATCH` in SLURM, `#PBS` in PBS).
**What it is for.** Running many ORCA calculations in parallel without saturating the machine; the scheduler decides when and where each one runs.
**In CADMIA.** **SLURM** support (local and SSH backends) is implemented; **PBS/Torque and SGE** are on the compatibility roadmap. See `cadmia prepare`/`cadmia submit`.
**Sources.** Yoo, Jette & Grondona, *SLURM*, in *Job Scheduling Strategies for Parallel Processing*, LNCS **2862**, 44 (2003).

---

## 12. Units, files and formats

### Energy units
- **Hartree (Eh):** atomic unit of energy. `1 Eh = 27.2114 eV = 627.5095 kcal/mol`.
- **eV (electronvolt):** common for orbital energies and gaps (HOMO/LUMO). `1 eV ≈ 23.06 kcal/mol`.
- **kcal/mol:** common for reaction/adsorption energies (ΔG, ΔH).
- **Debye (D):** dipole moment. `1 a.u. of dipole = 2.5417 D`.

### `.xyz`
**What it is.** A simple text format for geometries: number of atoms, comment line, and `symbol x y z` (in Å). CADMIA extracts clean geometries with `cadmia xyz`, preserving the atom order.

### `.cube` (Gaussian cube)
**What it is.** A file that stores a **3D scalar field** (density, MOs, MEP, Fukui) on a regular grid, plus the geometry. CADMIA reads it (`parse_cube`) for the maps.
**Key point.** To subtract cubes (Fukui, Δρ) they must all share **the same grid** (origin, dimensions, spacing).

### `.gbw`
**What it is.** An ORCA binary file with the wavefunction (orbitals and basis set). From it, `orca_plot` generates the `.cube` files of density/HOMO/LUMO/MEP.

### `.wfn`
**What it is.** A wavefunction format (orbitals as Gaussians) read by analyzers such as Multiwfn/AIM. CADMIA has a parser (`cadmia/io/wfn.py`).

---

## 13. Consolidated bibliography

**Fundamentals of DFT**
- Hohenberg, P.; Kohn, W. *Inhomogeneous Electron Gas.* Phys. Rev. **136**, B864 (1964). DOI: 10.1103/PhysRev.136.B864
- Kohn, W.; Sham, L. J. *Self-Consistent Equations Including Exchange and Correlation Effects.* Phys. Rev. **140**, A1133 (1965). DOI: 10.1103/PhysRev.140.A1133
- Parr, R. G.; Yang, W. *Density-Functional Theory of Atoms and Molecules.* Oxford University Press, 1989.

**Functionals and basis sets**
- Adamo, C.; Barone, V. *The PBE0 Model.* J. Chem. Phys. **110**, 6158 (1999). DOI: 10.1063/1.478522
- Weigend, F.; Ahlrichs, R. *Balanced Basis Sets (def2).* Phys. Chem. Chem. Phys. **7**, 3297 (2005). DOI: 10.1039/b508541a

**Conceptual DFT and reactivity**
- Geerlings, P.; De Proft, F.; Langenaeker, W. *Conceptual Density Functional Theory.* Chem. Rev. **103**, 1793 (2003). DOI: 10.1021/cr990029p
- Fukui, K. *Role of Frontier Orbitals in Chemical Reactions.* Science **218**, 747 (1982). DOI: 10.1126/science.218.4574.747
- Parr, R. G.; Yang, W. *Density Functional Approach to the Frontier-Electron Theory.* J. Am. Chem. Soc. **106**, 4049 (1984). DOI: 10.1021/ja00326a036
- Parr, R. G.; von Szentpály, L.; Liu, S. *Electrophilicity Index.* J. Am. Chem. Soc. **121**, 1922 (1999). DOI: 10.1021/ja983494x
- Morell, C.; Grand, A.; Toro-Labbé, A. *New Dual Descriptor for Chemical Reactivity.* J. Phys. Chem. A **109**, 205 (2005). DOI: 10.1021/jp046577a

**Acids and bases**
- Pearson, R. G. *Hard and Soft Acids and Bases.* J. Am. Chem. Soc. **85**, 3533 (1963). DOI: 10.1021/ja00905a001
- Parr, R. G.; Pearson, R. G. *Absolute Hardness: Companion Parameter to Absolute Electronegativity.* J. Am. Chem. Soc. **105**, 7512 (1983). DOI: 10.1021/ja00364a005
- Tanabe, K. *Solid Acids and Bases.* Academic Press, 1970.
- Lewis, G. N. *Valence and the Structure of Atoms and Molecules.* Chemical Catalog Co., 1923.

**Charges and bonds**
- Hirshfeld, F. L. *Bonded-Atom Fragments for Describing Molecular Charge Densities.* Theor. Chim. Acta **44**, 129 (1977). DOI: 10.1007/BF00549096
- Mayer, I. *Charge, Bond Order and Valence in the Ab Initio SCF Theory.* Chem. Phys. Lett. **97**, 270 (1983). DOI: 10.1016/0009-2614(83)80005-0

**Geometric structure**
- Hoppe, R. *Effective Coordination Numbers (ECoN).* Z. Kristallogr. **150**, 23 (1979). DOI: 10.1524/zkri.1979.150.14.23
- Cordero, B. et al. *Covalent Radii Revisited.* Dalton Trans. 2832 (2008). DOI: 10.1039/b801115j

**Methods and thermochemistry**
- Neese, F. *The ORCA Program System.* WIREs Comput. Mol. Sci. **2**, 73 (2012). DOI: 10.1002/wcms.81
- Neese, F. *Software Update: ORCA 5.0.* WIREs Comput. Mol. Sci. **12**, e1606 (2022). DOI: 10.1002/wcms.1606
- Bannwarth, C.; Ehlert, S.; Grimme, S. *GFN2-xTB.* J. Chem. Theory Comput. **15**, 1652 (2019). DOI: 10.1021/acs.jctc.8b01176
- Caldeweyher, E. et al. *Atomic-Charge Dependent London Dispersion Correction (D4).* J. Chem. Phys. **150**, 154122 (2019). DOI: 10.1063/1.5090222
- Grimme, S. *Supramolecular Binding Thermodynamics by Dispersion-Corrected DFT (Quasi-RRHO).* Chem. Eur. J. **18**, 9955 (2012). DOI: 10.1002/chem.201200497
- Luo, Y.-R. *Comprehensive Handbook of Chemical Bond Energies.* CRC Press, 2007.

**Catalysis and materials**
- Chorkendorff, I.; Niemantsverdriet, J. W. *Concepts of Modern Catalysis and Kinetics.* Wiley-VCH.
- Hammer, B.; Nørskov, J. K. *Theoretical Surface Science and Catalysis (d-band model).* Adv. Catal. **45**, 71 (2000). DOI: 10.1016/S0360-0564(02)45013-4
- Nørskov, J. K. et al. *Towards the Computational Design of Solid Catalysts.* Nat. Chem. **1**, 37 (2009). DOI: 10.1038/nchem.121
- Henrich, V. E.; Cox, P. A. *The Surface Science of Metal Oxides.* Cambridge University Press, 1994.

**Computing management (HPC)**
- Yoo, A. B.; Jette, M. A.; Grondona, M. *SLURM: Simple Linux Utility for Resource Management.* In *Job Scheduling Strategies for Parallel Processing*, LNCS **2862**, 44 (2003). DOI: 10.1007/10968987_3

**Wavefunction analysis**
- Lu, T.; Chen, F. *Multiwfn: A Multifunctional Wavefunction Analyzer.* J. Comput. Chem. **33**, 580 (2012). DOI: 10.1002/jcc.22885

---

*Living document. When adding a new term, follow the pattern: what it is · what it is for · equation (if applicable) · in CADMIA · real source. Scientific figures always come from the calculation, never from a language model.*
