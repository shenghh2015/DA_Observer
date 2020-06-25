#chcon -Rt svirt_sandbox_file_t /shared2/Data_FDA_Breast/Observer
docker run --gpus 0 -v /shared2/Data_FDA_Breast/Observer:/data -w /data/DA_Observer/FDA -it --user $(id -u):$(id -g) shenghh2020/tf_gpu_py3.5:latest bash
