# Protify

This is a Work In Progress repository which seeks to train a Transformer network that can craft a protein sequence which experiences similar intermolecular and interresidual forces to an inputted RNA sequence. This will then be able to leverage powerful protein-folding abilities to fold RNA molecules instead.

Here is a quick rundown of how to train this model for yourself:

1. You need a CUDA-enabled computer with Nvidia GPUs. We trained on Amazon EC2 G5 instances, using the Deep Learning with Tensorflow AMI.
2. Clone this repository with
   ```
    git clone https://github.com/jhwptsd/Protify.git
   ```
4. Open it
   ```
   cd Protify
   ```
5. Install libraries
   ```
   pip install .
   pip install colabfold[alphafold]
   pip install torch==2.0.1
   pip install "jax[cuda12_pip]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```
6. Run the code!
  ```
  python train_protify_local.py
  ```
