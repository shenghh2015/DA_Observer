# parser.add_argument("--gpu_num", type=int)
# parser.add_argument("--nb_cnn", type = int)
# parser.add_argument("--bn", type = bool)
# parser.add_argument("--lr", type = float)
# parser.add_argument("--nb_train", type = int)
# parser.add_argument("--bz", type = int)
# parser.add_argument("--optimizer", type = str)
# parser.add_argument("--nb_steps", type = int, default = 100000)
cd /home/shenghuahe/DA_Observer/CLB_FDA/v100_cluster
python train_target.py --gpu_num 0 --nb_cnn 4 --bn True --lr 1e-5 --nb_train 85000 --bz 400 --optimizer 'Adam' --nb_steps 5000
python target_target.py