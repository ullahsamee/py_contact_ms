<img width="784" height="442" alt="image" src="https://github.com/ullahsamee/py_contact_ms/blob/main/test/ms3.png" />

Longxing Cao's Contact Molecular Surface has been ported to python to allow the next generation of protein designers to use it with ease. Contact Molecular Surface (contact ms) is based on Lawrence and Colman's 1993 paper where they calculate Shape Complementarity. The difference is that instead of returning a singular value denoting the shape complementarity, contact ms instead returns a distance-weighted surface area of the target molecule.

At it's core, contact ms is based on the following formula:
`contact_ms = area * exp( -0.5 * distance**2)`

Where area is the interfacial area on the target and distance is the distance between the binder and the target (from the molecular surfaces) at that point.

Here's an image from (Brian) explaining why contact ms is better than SASA or Shape Complementarity:
<img width="784" height="442" alt="image" src="https://github.com/ullahsamee/py_contact_ms/blob/main/test/ms.png" />


In terms of using this library, there are really only two functions you need:
```python
from py_contact_ms import calculate_contact_ms, get_radii_from_names

# You'll have to figure out how to generate the following arrays
binder_xyz = xyz of binder heavy-atoms (non-hydrogen)
binder_res_names = list of residue name3 for each xyz (so like [ARG, ARG, ARG, LYS])
binder_atom_names = list of atom names for each xyz, stripped (so like [N, CA, C, O])
target_xyz = ...
target_res_names = ...
target_atom_names = ...

# Do not supply your own radii! CMS requires specific radii
binder_radii = get_radii_from_names(binder_res_names, binder_atom_names)
target_radii = get_radii_from_names(target_res_names, target_atom_names)

# Remember, contact_ms is only on the target side by convention
contact_ms, per_target_atom_cms, calc = calculate_contact_ms(binder_xyz, binder_radii, target_xyz, target_radii)

# If you also want the binder-side, you can do this (avoids recomputing everything)
binder_cms, per_binder_atom_cms = calc.calc_contact_molecular_surface(target_side=False)

# If you are doing small-molecule design, you may also want to know the maximum CMS possible (basically the surface area)
from py_contact_ms import calculate_maximum_possible_contact_ms
max_target_cms, max_target_cms_per_atom = calculate_maximum_possible_contact_ms(target_xyz, target_radii)
```

