from Utils import *

in_folder = "Validation/Actual"
out_folder = "Validation/Predicted/trRosettaRNA"

losses_RMSD = []
losses_TM = []

with torch.no_grad():
    for filename in os.listdir(in_folder):
        in_path = os.path.join(in_folder, filename)
        out_path = os.path.join(out_folder, filename)
        
        if os.path.isfile(in_path) and os.path.isfile(out_path):
            r1 = parse_rna(in_path, pdb=True)[0]
            r2 = parse_rna(out_path, pdb=True)[0]
            
            min_len = min(len(r1), len(r2))
            
            r1 = r1[:min_len]
            r2 = r2[:min_len]
            losses_RMSD.append(RMSD(r1, r2))
            losses_TM.append(-tm_score(r1, r2, len(r2)))
        
print(sum(losses_RMSD)/len(losses_RMSD))
print(sum(losses_TM)/len(losses_TM))

