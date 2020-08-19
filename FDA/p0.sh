## June 18
# python mmd_wd_check2.py --gpu 0 --docker True --dis_cnn 4 --g_cnn 4 --dis_fc 128 --g_lr 1e-5 --d_lr 1e-5 --lr 1e-5 --dis_bn True --den_bn True --iters 400000 --bz 100 --dis_param 1.0 --trg_clf_param 0 --mmd_param 0 --src_clf_param 1.0 --nb_trg_labels 0 --dataset total
# python mmd_adda.py --gpu 0 --docker True --dis_cnn 4 --dis_fc 128 --g_cnn 4 --g_lr 5e-6 --d_lr 5e-6 --lr 5e-6 --dis_bn True --den_bn True --iters 400000 --bz 200 --dis_param 1.0 --mmd_param 1.0 --trg_clf_param 0 --src_clf_param 1.0 --nb_trg_labels 0 --dataset total --drop 0.5
# parser.add_argument("--gpu", type=int, default = 2)
# parser.add_argument("--docker", type = str2bool, default = True)
# parser.add_argument("--d_cnn", type = int, default = 0)
# parser.add_argument("--g_cnn", type=int, default = 6)
# parser.add_argument("--fc", type=int, default = 256)
# parser.add_argument("--d_bn", type=str2bool, default = False)
# parser.add_argument("--g_bn", type=str2bool, default = False)
# parser.add_argument("--c_lr", type=float, default = 1e-5)
# parser.add_argument("--g_lr", type=float, default = 1e-5)
# parser.add_argument("--d_lr", type=float, default = 1e-5)
# parser.add_argument("--bz", type = int, default = 100)
# parser.add_argument("--itr", type = int, default = 10000)
# parser.add_argument("--d_weight", type = float, default = 1.0)
# parser.add_argument("--t_weight", type = float, default = 1.0)
# parser.add_argument("--s_weight", type = float, default = 1.0)
# parser.add_argument("--labels", type = int, default = 0)
# parser.add_argument("--dataset", type = str, default = 'total')

python mmd_fake.py --gpu 0 --g_cnn 6 --fc 256 --c_lr 1e-4 --bz 300 --itr 100000 --d_weight 0.1 --s_weight 1.0 --pseudo_label True