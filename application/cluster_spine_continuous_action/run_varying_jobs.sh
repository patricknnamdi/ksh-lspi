for DF in 10; \
do for H in 0.001 0.005 0.01; \
do for mu in 0.0001; \
do for lambda in 0 1 10 100; \
do for degree in 3; \
do for discount in 0 0.5 0.8 0.99; \
do for policy_iter in 1 2; do
#
echo "DF: ${DF}, H: ${H}, degree: ${degree}, mu: ${mu}, lambda: ${lambda}, discount: ${discount}, policy_iter: ${policy_iter}"
export DF H degree mu lambda discount policy_iter
#
sbatch -o out_df${DF}_h${H}_degree${degree}_mu${mu}_lambda${lambda}_discount${discount}_policy_iter${policy_iter}.stdout.txt \
-e out_err_df${DF}_h${H}_degree${degree}_mu${mu}_lambda${lambda}_discount${discount}_policy_iter${policy_iter}.stdout.txt \
--job-name=my_analysis_discount_${discount} \
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