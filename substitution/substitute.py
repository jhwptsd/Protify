from parse_proteins import parse_protein
from Monomer import Monomer
import random

def substitute(rna, protein):#, path_out):
    points, norms = parse_protein(protein)
    out = ""
    length = 1
    for i in range(len(rna)):
        temp = Monomer("nucleotide")
        temp.load_template(rna[i])
        temp.add_name(rna[i])
        # TODO: Find out how to rotate the nucleotides correctly to match the amino acids
        temp = temp.align_to_normal(norms[i])
        temp.apply_transformation(points[i][0], points[i][1], points[i][2])
        out += temp.print(start=length, number=i+1)
        length = length + len(temp)
    
    return out
        
ns = {0:'A', 1:'U', 2:'G', 3:'C'}
seq = []
for i in range(68):
    seq.append(ns[random.randint(0,3)])

s = substitute(seq, "Proteintest.cif")

f = open("generated_test.cif", "w")
f.write("""data_generated-sequence
#
loop_
_atom_site.group_PDB 
_atom_site.id 
_atom_site.type_symbol 
_atom_site.label_atom_id 
_atom_site.label_alt_id 
_atom_site.label_comp_id 
_atom_site.label_asym_id 
_atom_site.label_entity_id 
_atom_site.label_seq_id 
_atom_site.pdbx_PDB_ins_code 
_atom_site.Cartn_x 
_atom_site.Cartn_y 
_atom_site.Cartn_z
""")
f.write(s)
f.close()