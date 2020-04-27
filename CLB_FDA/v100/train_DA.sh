cd /home/shenghuahe/DA_Observer/CLB_FDA/v100
python2 job_parser.py 'DA_jobs.sh'
for i in $(seq 0 3)
do
   sh job_folder/job_$i.sh&
   sleep 120s &
done
wait
