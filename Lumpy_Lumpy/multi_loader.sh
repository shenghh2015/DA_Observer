cd /home/shenghuahe/DA_Observer/Lumpy_Lumpy
python2 v100_job_parser.py 'multi_GPU.sh'
for i in $(seq 0 1)
do
   sh v100_jobs/job_$i.sh&
   sleep 10s &
done
wait
