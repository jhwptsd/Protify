from Utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

in_folder = "Validation/Actual"
out_folder = "Validation/Predicted/AlphaFold"

cif = True

losses_RMSD = []
losses_TM = []
lengths = []

with torch.no_grad():
    for filename in os.listdir(in_folder):
        in_path = os.path.join(in_folder, filename)
        out_path = os.path.join(out_folder, filename)

        if cif:
            out_path = out_path.replace(".pdb", ".cif")
        
        if os.path.isfile(in_path) and os.path.isfile(out_path):
            r1 = parse_rna(in_path, pdb=True)[0]
            r2 = parse_rna(out_path, pdb=not cif)[0]
            
            min_len = min(len(r1), len(r2))
            
            r1 = r1[:min_len]
            r2 = r2[:min_len]
            lengths.append(min_len)
            r1, r2 = kabsch_algorithm(r1, r2)
            losses_RMSD.append(RMSD(r1, r2))
            losses_TM.append(-tm_score(r1, r2, len(r2)))
        
print(sum(losses_RMSD)/len(losses_RMSD))
print(sum(losses_TM)/len(losses_TM))

log_lengths = np.log(lengths)
m, c = np.polyfit(log_lengths, losses_RMSD, 1)

x_fit = np.linspace(np.min(lengths), np.max(lengths), 100)
y_fit = m * np.log(x_fit) + c

fig, ax = plt.subplots(1, 1, layout='constrained')
ax.scatter(lengths, losses_RMSD, label='Data')
ax.plot(x_fit, y_fit, 'r-', label='Logarithmic fit')
ax.set_xlabel("Length (bps)")
ax.set_ylabel("Loss")
ax.legend()
plt.show()

