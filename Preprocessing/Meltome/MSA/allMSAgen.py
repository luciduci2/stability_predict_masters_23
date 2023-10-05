#! /home/lucas/miniconda3/envs fresh
""" This script starts subprocesses that in turn start generating MSAs with hh-blits
By default this script runs on 96 scripts, change numBatches and num_CPU if you do not have access to this number of CPUs. 
It is recommended to run this script in the background with nohup.
By default this script requires the directory .../Meltome/seqdata/ with FASTA sequences for the proteins as well as the text-file 
.../Meltome/seqdata/index.txt with the file-names to be aligned. By default it requires the directory .../Meltome/MSA/MSAs to place the generated MSAS.

input numBatches: number of subprocesses
input indexFile: a textfile containing the filenames of the FASTA-files that will be aligned.
input dirInName: the directory of the file with the name described in indexFile can be found.
input dirOutName: the directory where the MSAs should be saved.
input n: how many iterations the MSA should be created for. An input for hh-blits.
input num_CPU: number of CPUs each MSA generation should utilise.
input print: Should CH-HIT print its output.
input dirLib: the directory where the sequence library is located,
input script_path: the path of the script that calls hh-blits.
"""

import subprocess
import sys
import os
import shutil

def main(numBatches=16, 
         indexFile='Preprocessing/Meltome/seqdata/index.txt', 
         dirInName= '/home/lucas/Protein_Stability_Masters_2023/Preprocessing/Meltome/seqdata/',
         dirOutName = '/home/lucas/Protein_Stability_Masters_2023/Preprocessing/Meltome/MSA/MSAs/',
         n=3,
         num_CPU=4,
         print=True,
         dirLib='/home/vera/projects/masters_project/tests/hhsuite-3.3.0-AVX2-Linux/databases/uniclust30_2018_08/uniclust30_2018_08',
         script_path='Preprocessing/Meltome/MSA/batchMSAgen.py'):
    

    if not os.path.exists(dirOutName):
        os.mkdir(dirOutName)

    with open(indexFile) as input_file:
        indices = input_file.read().splitlines()

    batch_list = [indices[i::numBatches] for i in range(numBatches)]

    for i in range(numBatches):
        command = ['python', script_path, '+'.join(batch_list[i]), dirInName, dirOutName, str(n), str(num_CPU), str(print), dirLib]
        subprocess.Popen(command, env=os.environ)

if __name__ == "__main__":
    main(16)