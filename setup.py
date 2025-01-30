import os

os.system("pip install .")
os.system("pip install colabfold[alphafold]")
os.system("pip install torch==2.0.1")
os.system('pip install "jax[cuda12_pip]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html')