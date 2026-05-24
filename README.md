<img width="784" height="442" alt="image" src="https://github.com/ullahsamee/py_contact_ms/blob/main/test/ms3.png" />

Longxing Cao's Contact Molecular Surface has been ported to python to allow protein designers to use it with ease. Contact Molecular Surface (contact ms) is based on [Lawrence & Colman's 1993 Shape Complementarity paper](https://doi.org/10.1006/jmbi.1993.1648) paper where they calculate Shape Complementarity. 

## What is Contact Molecular Surface? 
Classic **Shape Complementarity (SC)** returns a single scalar value describing how well two molecular surfaces fit together. Classic **Delta SASA** measures how much solvent-accessible surface area is buried upon binding. Both are useful, but both have blind spots.
 
**Contact MS fixes those blind spots.**
 
Instead of a single scalar, CMS returns a **distance-weighted surface area on the target molecule** — rewarding tight, close-contact interfaces and penalizing regions where a gap exists between binder and target, even if those regions are buried (and therefore counted by Delta SASA) or geometrically matched (and therefore counted by SC).
 
### Core Formula
 
```
contact_ms = area × exp(−0.5 × distance²)
```
 
Where:
- **`area`** — the interfacial area element on the target's molecular surface
- **`distance`** — the gap between the binder and target molecular surfaces at that point

---
 
## Why Not Just Use SASA or Shape Complementarity?
 
The figure below (from Brian) illustrates four interface scenarios of increasing quality (left → right), and shows precisely where SC and Delta SASA break down:
 
Here's an illustration from Brian explaining why contact ms is better than SASA or Shape Complementarity
<img width="784" height="442" alt="image" src="https://github.com/ullahsamee/py_contact_ms/blob/main/test/ms.png" />
 
| Scenario | Gap Size | SASA | SC | Contact MS |
|---|---|---|---|---|
| 1 — Gap larger than SASA probe | Very large | Low | High | **Low** ✓ |
| 2 — Gap slightly smaller than SASA probe (~2.2 Å) | ~2.2 Å | High | High | **Low** ✓ |
| 3 — Gap ~1.0 Å | ~1.0 Å | High | Low | **Medium** ✓ |
| 4 — Gap very small | Near zero | High | High | **High** ✓ |

**The critical failure cases:**
 
- **Scenarios 2 vs 4 are indistinguishable under both SASA and SC** — both report High/High — yet scenario 4 is a far superior interface. Contact MS correctly ranks 4 above 2.
- **SC ignores the gap region in scenario 2** entirely (the SC calculation region excludes the poorly-packed area), so it never penalizes the gap.
- **Delta SASA counts buried area regardless of contact quality** — a wide gap that happens to be buried still looks good to SASA.
 
Contact MS correctly penalizes gaps via the exponential term, producing a metric that tracks true interface quality.

**Installation**
```bash
pip install py-contact-ms
```
## Recommended Installation (Virtual Environment)

```bash
git clone https://github.com/ullahsamee/py_contact_ms.git
cd py_contact_ms

python3 -m venv venv
source venv/bin/activate

pip install -e .
```

## Usage


The library exposes two primary functions. Here is a complete working example:
 
```python
from py_contact_ms import calculate_contact_ms, get_radii_from_names
 
# ── Inputs ──────────────────────────────────────────────────────────────────
# All arrays are over heavy atoms (non-hydrogen) only.
 
binder_xyz        = ...  # (N, 3) array of binder atom coordinates
binder_res_names  = ...  # list of 3-letter residue names per atom, e.g. ["ARG", "ARG", "LYS"]
binder_atom_names = ...  # list of stripped atom names per atom,   e.g. ["N", "CA", "C", "O"]
 
target_xyz        = ...
target_res_names  = ...
target_atom_names = ...
 
# ── Radii ────────────────────────────────────────────────────────────────────
# ⚠️  Always use get_radii_from_names — CMS requires its own specific radii.
#     Do NOT substitute your own van der Waals or other radii sets.
 
binder_radii = get_radii_from_names(binder_res_names, binder_atom_names)
target_radii = get_radii_from_names(target_res_names, target_atom_names)
 
# ── Target-side CMS (the standard convention) ────────────────────────────────
contact_ms, per_target_atom_cms, calc = calculate_contact_ms(
    binder_xyz, binder_radii,
    target_xyz, target_radii,
)
# contact_ms            → scalar total CMS value for the target side
# per_target_atom_cms   → (M,) array, per-atom contribution on the target
# calc                  → calculator object for further queries (avoids recompute)
 
# ── Binder-side CMS (optional, reuses the same calculation) ─────────────────
binder_cms, per_binder_atom_cms = calc.calc_contact_molecular_surface(target_side=False)
 
# ── Maximum possible CMS (useful for small-molecule design) ──────────────────
# Returns what CMS would be if the target surface were perfectly contacted
# everywhere — essentially a normalised surface area upper bound.
from py_contact_ms import calculate_maximum_possible_contact_ms
 
max_target_cms, max_target_cms_per_atom = calculate_maximum_possible_contact_ms(
    target_xyz, target_radii
)
```
 
### Quick Reference
 
| Function | Returns | When to use |
|---|---|---|
| `get_radii_from_names(res_names, atom_names)` | radii array | Always — use this instead of your own radii |
| `calculate_contact_ms(binder_xyz, binder_radii, target_xyz, target_radii)` | `(scalar, per_atom_array, calc_obj)` | Standard binder–target interface scoring |
| `calc.calc_contact_molecular_surface(target_side=False)` | `(scalar, per_atom_array)` | Binder-side CMS without recomputing |
| `calculate_maximum_possible_contact_ms(xyz, radii)` | `(scalar, per_atom_array)` | Small-molecule design; CMS normalisation |
 
---
 
## Key Conventions
 
- **Target-side by default.** CMS is reported on the target molecule by convention. Use `target_side=False` if you need the binder-side score.
- **Heavy atoms only.** Strip all hydrogen atoms before passing coordinates.
- **Use the bundled radii.** The exponential distance weighting is calibrated to a specific radii set; substituting other radii will produce meaningless values.
- **Units.** Area is in Å², distance in Å; the output CMS has units of Å².
 
---

## Example: Protein–Protein Complex (run_cms.py) <-- save file and run.
I assume:
Chain A = target receptor
Chain B = binder
```python
import numpy as np
from Bio.PDB import PDBParser
from py_contact_ms import calculate_contact_ms
from py_contact_ms import get_radii_from_names

pdb_file = "design_complex.pdb"

parser = PDBParser(QUIET=True)
structure = parser.get_structure("complex", pdb_file)

binder_xyz = []
binder_res = []
binder_atoms = []

target_xyz = []
target_res = []
target_atoms = []

for atom in structure.get_atoms():
    residue = atom.get_parent()
    chain = residue.get_parent().id

    coord = atom.coord
    resname = residue.resname
    atomname = atom.name.strip()

    if atom.element == "H" or atomname.startswith("H") or (len(atomname) > 1 and atomname[0].isdigit() and atomname[1] == "H"):
        continue

    # Binder peptide
    if chain == "B":
        binder_xyz.append(coord)
        binder_res.append(resname)
        binder_atoms.append(atomname)

    # Target receptor
    elif chain == "A":
        target_xyz.append(coord)
        target_res.append(resname)
        target_atoms.append(atomname)

binder_xyz = np.array(binder_xyz)
target_xyz = np.array(target_xyz)

binder_radii = get_radii_from_names(
    binder_res,
    binder_atoms
)

target_radii = get_radii_from_names(
    target_res,
    target_atoms
)

contact_ms, per_atom_cms, calc = calculate_contact_ms(
    binder_xyz,
    binder_radii,
    target_xyz,
    target_radii
)

print("CMS Score:", contact_ms)
```

```bash
===============================
Contact Molecular Surface (CMS)
===============================
CMS Score: 427.02491532312615
```
