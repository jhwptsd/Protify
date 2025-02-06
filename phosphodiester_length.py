from Utils import *
from tqdm import tqdm
# Instead of regressing to find tgt lengths, we can just use the average length of the phosphodiester bond

seqs = {} # All sequences - may get quite large

# Used for file tree searching
components = []
macro_tags = []


def get_structure(tag, path):
        # Return the structure of an RNA molecule given its tag and the path to the structure directory
        # File directory:
        # root
        #  -- colabfold.dir
        #  -- train_protify.ipynb
        #  -- data.dir
        #  ---- component 1.dir
        #  ------ tag 1.dir
        #  -------- tag 1a.cif
        #  -------- tag 1b.cif
        # ...
        index = list(old_seqs.keys()).index(tag)
        component = components[index]
        macro_tag = macro_tags[index]

        path = f"{path}/train_set/{component}/{macro_tag}/{tag}.cif"
        return path

if __name__=="__main__":

    if not os.path.exists("rna3db-mmcifs"):
        print("Downloading RNA3Db...")
        # Donwload RNA3Db structure files
        os.system('wget https://github.com/marcellszi/rna3db/releases/download/incremental-update/rna3db-mmcifs.v2.tar.xz')
        print("Extracting structure files...")
        os.system('sudo tar -xf rna3db-mmcifs.v2.tar.xz')
        os.system('rm rna3db-mmcifs.v2.tar.xz')

        # Donwload RNA3Db sequence files
        os.system('wget https://github.com/marcellszi/rna3db/releases/download/incremental-update/rna3db-jsons.tar.gz')
        print("Extracting sequence files...")
        os.system('tar -xzf rna3db-jsons.tar.gz')
        os.system('rm rna3db-jsons.tar.gz')
    seq_path = "rna3db-jsons/cluster.json"
    struct_path = "rna3db-mmcifs"

    # Load up the data
    old_seqs, components, macro_tags = load_data(seq_path, 0, 1645, max_len=100)
    lengths = []
    
    # Parse sequences and get the average phosphodiester bond length
    for i in tqdm(range(len(old_seqs))):
         pts = parse_rna(get_structure(list(old_seqs.keys())[i], struct_path))[0].detach().cpu().numpy()
         temp_lengths = []
         for i in range(len(pts)-1):
             temp_lengths.append(np.linalg.norm(pts[i]-pts[i+1]))
         temp_lengths = np.mean(np.array(temp_lengths))
         lengths.append(temp_lengths)
    lengths = np.array(lengths)
    print(np.mean(lengths), np.std(lengths))