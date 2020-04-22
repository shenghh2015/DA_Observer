FROM tensorflow/tensorflow:1.13.0rc1-gpu-py3

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends python3-pip

RUN pip install scipy sklearn matplotlib natsort ipython

WORKDIR /data

## 
#  1. build up a docker container
#  docker build -f Dockerfile -t shenghh2020/tf_gpu_py3.5:latest -t shenghh2020/tf_gpu_py3.5:1.0
#  2. push the docker container to the docker hub
#  docker push shenghh2020/tf_gpu_py3.5:latest shenghh2020/tf_gpu_py3.5:1.0
#  3. qsub a job to the v100_cluster
#  bsub -Is -G compute-anastasio -q anastasio-interactive -a 'docker(shenghh2020/tf_gpu_py3.5:latest)' -gpu "num=4" /bin/bash
#  when the access permission is required, use the following command:
#
# docker login -u "myusername" -p "mypassword" docker.io
# docker push myusername/myimage:0.0.1
##
