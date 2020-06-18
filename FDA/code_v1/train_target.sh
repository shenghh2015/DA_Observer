# parser.add_argument("--gpu_num", type=int)
# parser.add_argument("--lr", type = float)
# parser.add_argument("--nb_train", type = int)
# parser.add_argument("--bz", type = int)
# parser.add_argument("--optimizer", type = str)

python train_target.py --gpu_num 1 --lr 1e-5 --nb_train 85000 --bz 400 --optimizer 'Adam'
python train_target1.py --gpu_num 2 --lr 1e-5 --nb_train 85000 --bz 400 --optimizer 'Adam'