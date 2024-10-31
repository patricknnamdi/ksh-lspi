for STATE in 0 1 2 3; \
do for DF in 10; \
do for H in 0.01 0.05; \
do for mu in 0.0001; \
do for lambda in 0 1 10 100; \
do for degree in 3; \
do for discount in 0 0.5 0.8 0.99; \
do for policyt_iter in 1 2; do
#
echo "STATE: ${STATE}, DF: ${DF}, H: ${H}, degree: ${degree}, mu: ${mu}, lambda: ${lambda}, discount: ${discount}, policy_iter: ${policy_iter}"
export STATE DF H degree mu lambda discount policy_iter
#
sbatch -o out_state${STATE}_df${DF}_h${H}_degree${degree}_mu${mu}_lambda${lambda}_discount${discount}_policy_iter${policy_iter}.stdout.txt \
-e out_err_state${STATE}_df${DF}_h${H}_degree${degree}_mu${mu}_lambda${lambda}_discount${discount}_policy_iter${policy_iter}.stdout.txt \
--job-name=my_analysis_discount_state${STATE}_${discount} \
shell_kshlspi_reg.sh
#exit
sleep 0.2 # pause to be kind to the scheduler
done
done
done
done
done
done
done
done