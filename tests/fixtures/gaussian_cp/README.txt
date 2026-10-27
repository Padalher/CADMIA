counterpoise_synthetic.log: synthetic Gaussian counterpoise=2 output.

It keeps only the lines the parser reads. The complex comes first and the last "SCF Done"
belongs to a fragment, so the energy must come from "Counterpoise: corrected energy =", never
from the last SCF Done.

Expected values:
  energy (corrected)  -1001.408680229182 Ha
  BSSE                    0.001604618594 Ha
  last SCF Done         -500.30298180 Ha   (a fragment; must NOT be returned)

1tial_CO2_cp.out: real Gaussian 16 Counterpoise=2 single point of the Al2TiO5 + CO2 adduct
(PBE1PBE/def2TZVP, fragment 1 = cluster, fragment 2 = CO2), at the geometry of
1tial_d4/1tial_CO2.out. Host and paths replaced. Its five SCF energies give the BSSE it prints:
  corrected energy  -1898.673359045103 Ha  = complex -1898.67578709 + BSSE
  BSSE                  0.002428048502 Ha  (1.5236 kcal/mol)
