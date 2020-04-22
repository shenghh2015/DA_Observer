# parser.add_argument("--gpu", type=int)
# parser.add_argument("--dis_cnn", type=int)
# parser.add_argument("--dis_fc", type=int)
# parser.add_argument("--dis_bn", type=bool)
# parser.add_argument("--D_lr", type = float)
# parser.add_argument("--G_lr", type = float)
# parser.add_argument("--nD", type = int)
# parser.add_argument("--nG", type = int)
# parser.add_argument("--dAcc1", type = int)
# parser.add_argument("--dAcc2", type = int)
# parser.add_argument("--iters", type = int)
# parser.add_argument("--bz", type = int)

python train_DA.py --gpu 6 --dis_cnn 2 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 2 --nG 2 --iters 10000 --bz 200
python train_DA.py --gpu 6 --dis_cnn 2 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 2 --nG 1 --iters 10000 --bz 200
python train_DA.py --gpu 0 --dis_cnn 4 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 2 --nG 1 --iters 10000 --bz 200
python train_DA.py --gpu 3 --dis_cnn 2 --dis_bn True --D_lr 1e-4 --G_lr 1e-5 --nD 1 --nG 1 --iters 200000 --bz 200
python train_DA.py --gpu 3 --dis_cnn 2 --dis_fc 128 --dis_bn False --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --iters 200000 --bz 400
python train_DA.py --gpu 3 --dis_cnn 2 --dis_fc 512 --dis_bn True --D_lr 1e-4 --G_lr 1e-4 --nD 1 --nG 1 --dAcc1 0.8 --dAcc2 0.9 --iters 200000 --bz 400
python train_DA.py --gpu 6 --dis_cnn 2 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.8 --dAcc2 0.9 --iters 200000 --bz 400
python train_DA.py --gpu 0 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 2 --nG 1 --dAcc1 0.8 --dAcc2 0.9 --iters 10000 --bz 200
python train_DA.py --gpu 0 --dis_cnn 4 --dis_fc 256 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.8 --dAcc2 0.9 --iters 10000 --bz 400
