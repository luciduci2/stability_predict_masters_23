#! /home/lucas/miniconda3/envs fresh

import subprocess
import sys
import os


def main(batchIndices, dirIn, dirOut, n, cpu, print, dirLib):

    if not os.path.exists(dirOut):
        print('ERROR: Cannot find directory for aligned output.')
       
    indices = batchIndices.split('+')
    for id in indices:
        PATH_In = dirIn + id
        PATH_Out = dirOut+ id[:-6] + '.a3m'
        line = 'hhblits -i '+PATH_In+' -oa3m '+PATH_Out+' -n '+str(n)+' -d '+dirLib+' -cpu '+ str(cpu)
        
        if os.path.exists(PATH_Out):
            sys.stdout.write('INFO: MSA already found in output directory for ' + id + '\n')
            continue
        if not os.path.exists(PATH_In):
            sys.stdout.write('ERROR: Cannot find the input sequence for ' + id + ' in path: ' + PATH_In + '\n')
            continue
        try:
            
            subprocess.run([line], shell=True, text=True, check=True)
        except subprocess.CalledProcessError:
            sys.stdout.write('ERROR: Did not manage to produce MSA for ' + id + '.\n')
            continue


if __name__ == "__main__":
    #main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    main('P19079_WT_.FASTA+O25103_WT_.FASTA', 'Preprocessing/ProThermDB/seqdata/' ,'Preprocessing/ProThermDB/MSA/MSAs/', '3', '4', 'True', '/home/vera/projects/masters_project/tests/hhsuite-3.3.0-AVX2-Linux/databases/uniclust30_2018_08/uniclust30_2018_08')