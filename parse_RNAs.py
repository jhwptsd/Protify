import sys
from substitution.Monomer import *
from gemmi import *
import numpy as np
import torch


def parse_rna(path):
    try:
        doc = cif.read_file(path)
        block = doc.sole_block()
        all_xs = [element for element in block.find_loop("_atom_site.Cartn_x")]
        all_ys = [element for element in block.find_loop("_atom_site.Cartn_y")]
        all_zs = [element for element in block.find_loop("_atom_site.Cartn_z")]
        all_atoms = [element for element in block.find_loop("_atom_site.label_atom_id")]
        
        points = []
        angle_points = []
        norms = []
        
        for x, y, z, atom in zip(all_xs, all_ys, all_zs, all_atoms):
            x = float(x)
            y = float(y)
            z = float(z)

            if atom == "P":
                points.append(np.array([x, y, z]))
                angle_points.append(np.array([x, y, z]))
            elif atom == "\"C1'\"":
                angle_points.append(np.array([x, y, z]))
            elif atom == "\"C4'\"":
                angle_points.append(np.array([x, y, z]))
                v1 = angle_points[-1]-angle_points[-2]
                v2 = angle_points[-3]-angle_points[-2]
                norms.append(np.cross(v1, v2))
                angle_points = []
        
        return torch.Tensor(points), torch.Tensor(norms)
            
    except Exception as e:
        print("Oops. %s" % e)
        sys.exit(1)