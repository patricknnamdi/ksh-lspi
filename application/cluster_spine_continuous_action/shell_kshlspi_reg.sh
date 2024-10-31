#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-15:00 
#SBATCH --mem=300         
#SBATCH -p serial_requeue        
#SBATCH -e hostname_%j.err   
#SBATCH --mail-type=FAIL 
#SBATCH --mail-user=patrickemedom@g.harvard.edu

module load python/3.7.7-fasrc01
source activate ksh_env

python3 kshlspi_cluster_reg.py ${STATE} ${DF} ${H} ${degree} ${mu} ${lambda} ${discount} ${policy_iter}