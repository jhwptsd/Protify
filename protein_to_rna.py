from loss import RMSD, tm_score
from parse_RNAs import parse_rna
from parse_proteins import parse_protein
import numpy as np

def protein_to_rna(protein, rna_path, tm=False):
    prot_points, _ = parse_protein(protein)
    rna_points, _ = parse_rna(rna_path)
    prot_points = correct_protein_coords(prot_points, len(prot_points))
    if tm:
        return tm_score(prot_points, rna_points)
    else:
        return RMSD(prot_points, rna_points)
    
def correct_protein_coords(points, points_length, correction_factor=np.array([0,0,0]), new_pts=np.array([])):
    # Apply correction factor to the protein coordinates to account for bond lengths
    pp_dist = 6.8 # Approximated this value from what ChatGPT tells me - will look for rigorous results
                  # Also haha pp
    if len(new_pts)==points_length:
        return new_pts
    v = np.array(points[1])-np.array(points[0]) # Vector between two points
    v = correction_factor + (np.linalg.norm(v)-pp_dist)*v/np.linalg.norm(v) # Delta correction factor
    new = new_pts
    if correction_factor==np.array([0,0,0]):
        new.append(np.array(points[0])) # Add first point on the outermost function call
    new.append(np.array(points[1])-v) # Apply correction factor to next point and add
    return correct_protein_coords(points[1:], points_length, v, new) # Recursive call