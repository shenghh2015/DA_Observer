# parser.add_argument("--gpu", type=int)
# parser.add_argument("--D_lr", type = float)
# parser.add_argument("--G_lr", type = float)
# parser.add_argument("--nD", type = int)
# parser.add_argument("--nG", type = int)
# parser.add_argument("--iters", type = int)
# parser.add_argument("--bz", type = int)

python train_DA.py --gpu 0 --D_lr 1e-5  --G_lr 1e-6 --nD 5 --nG 5 --bz 400 --iters 100
