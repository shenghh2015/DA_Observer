cd /scratch/00-VICTRE_pipeline/01-pipeline_codes/x-ray_runs
for i in $(seq 0 3)
do
   python run_SP_jobs.py --num_jobs 100 --gpu_num $i&
   sleep 120s &
#   sleep $[($RANDOM % 40)+10]s &
done
wait

