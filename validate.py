from Utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

in_folder = "Validation/Actual"
out_folder = "Validation/Predicted/trRosettaRNA"

losses_RMSD = []
losses_TM = []
lengths = []

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
            lengths.append(min_len)
            losses_RMSD.append(RMSD(r1, r2))
            losses_TM.append(-tm_score(r1, r2, len(r2)))
        
print(sum(losses_RMSD)/len(losses_RMSD))
print(sum(losses_TM)/len(losses_TM))


fig, ax = plt.subplots(1, 1, layout='constrained')
ax.scatter(lengths, losses_RMSD)

coeffs = np.polyfit(lengths, losses_RMSD, 1)
poly_eq = np.poly1d(coeffs)
x_smooth = np.linspace(min(lengths), max(lengths), 100)
ax.plot(x_smooth, poly_eq(x_smooth), 'g-', label="Polynomial Fit")

ax.set_xlabel("Length (bps)")
ax.set_ylabel("Loss (RMSD)")
plt.show()

