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

