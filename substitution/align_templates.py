import sys
from Monomer import *
from gemmi import *

atoms = []
atom_xs = []
atom_ys = []
atom_zs = []
for path in sys.argv[1:]:
    try:
        doc = cif.read_file(path)  # copy all the data from mmCIF file
        block = doc.sole_block()  # mmCIF has exactly one block
        for element in block.find_loop("_atom_site.label_atom_id"):
            atoms.append(element)
        for element in block.find_loop("_atom_site.Cartn_x"):
            atom_xs.append(float(element))
        for element in block.find_loop("_atom_site.Cartn_y"):
            atom_ys.append(float(element))
        for element in block.find_loop("_atom_site.Cartn_z"):
            atom_zs.append(float(element))
    except Exception as e:
        print("Oops. %s" % e)
        sys.exit(1)
        
m = Monomer("nucleotide")
m.add_name("A")
for i in range(len(atoms)):
    m.add_atom(atom_xs[i], atom_ys[i], atom_zs[i], atoms[i])

i_p = atoms.index("P")

# Center the atom's P molecule at (0, 0, 0)
m = m.apply_transformation(-atom_xs[i_p], -atom_ys[i_p], -atom_zs[i_p])
m = m.align_triangle_to_xy()

print(m)