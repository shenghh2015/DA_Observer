cd /home/sh38/DA_Observers/CLB_FDA
# May 3, 2020
#python train_target.py --gpu_num 0 --docker False nb_cnn 4 --bn False --lr 1e-5 --nb_train 70 --bz 50 --nb_steps 25000 --optimizer Adam --dataset total
#python train_target.py --gpu_num 0 --docker False nb_cnn 4 --bn False --lr 1e-5 --nb_train 100--bz 50 --nb_steps 25000 --optimizer Adam --dataset total
python train_target.py --gpu_num 1 --docker False --nb_cnn 4 --bn False --lr 1e-5 --nb_train 200 --bz 100 --nb_steps 25000 --optimizer Adam --dataset total
python train_target.py --gpu_num 1 --docker False --nb_cnn 4 --bn False --lr 1e-5 --nb_train 300 --bz 100 --nb_steps 25000 --optimizer Adam --dataset total
# python train_target.py --gpu_num 0 --docker False nb_cnn 4 --bn False --lr 1e-5 --nb_train 400 --bz 50 --nb_steps 25000 --optimizer Adam --dataset total

# parser.add_argument("--gpu_num", type=int)
# parser.add_argument("--docker", type = str2bool, default = True)
# parser.add_argument("--nb_cnn", type = int)
# parser.add_argument("--bn", type = str2bool, default = False)
# parser.add_argument("--lr", type = float)
# parser.add_argument("--nb_train", type = int)
# # parser.add_argument("--noise", type = float)
# # parser.add_argument("--sig_rate", type = float)
# parser.add_argument("--bz", type = int)
# parser.add_argument("--optimizer", type = str)
# parser.add_argument("--nb_steps", type = int, default = 100000)
# parser.add_argument("--dataset", type = str, default = 'total')
