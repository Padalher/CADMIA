counterpoise_synthetic.log: synthetic Gaussian counterpoise=2 output.

It keeps only the lines the parser reads. The complex comes first and the last "SCF Done"
belongs to a fragment, so the energy must come from "Counterpoise: corrected energy =", never
from the last SCF Done.

Expected values:
  energy (corrected)  -1001.408680229182 Ha
  BSSE                    0.001604618594 Ha
  last SCF Done         -500.30298180 Ha   (a fragment; must NOT be returned)
