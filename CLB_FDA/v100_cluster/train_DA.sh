cd /home/shenghuahe/DA_Observer/CLB_FDA/v100_cluster
python job_parser.py
for i in $(seq 0 3)
do
   sh job_folder/job_$i.sh&
#    sleep 60s &
done
wait
