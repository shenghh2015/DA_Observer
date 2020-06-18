LSF_DOCKER_VOLUMES='/scratch1/fs1/anastasio/Data_FDA_Breast/DA_Observer:/data' LSF_DOCKER_NETWORK=host LSF_DOCKER_IPC=host LSF_DOCKER_SHM_SIZE=40G bsub -G compute-anastasio -n 4 -R 'span[ptile=4] select[mem>200000] rusage[mem=200GB]' -Is -q general-interactive -a 'docker(shenghh2020/tf_gpu_py3.5:latest)' -gpu "num=4" -o /scratch1/fs1/anastasio/Data_FDA_Breast/DA_Observer/logs/DA_$RANDOM /bin/bash