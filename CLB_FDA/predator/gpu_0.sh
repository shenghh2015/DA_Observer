# python train_source.py --gpu_num 0 --nb_cnn 6 --bn True --lr 1e-5 --nb_train 100000 --noise 2 --sig_rate 0.035 --bz 400 --optimizer Adam --nb_steps 10000
# python mmd_DA.py --gpu 0 --docker True --shared True --lr 1e-5 --iters 100000 --bz 400 --mmd_param 0.5 --nb_trg_labels 0
# python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 100000 --bz 400 --nb_trg_labels 500 --mmd_param 1.0 --trg_clf_param 1.0 --src_clf_param 1.0
# python TF.py --gpu 0 --docker True --lr 1e-7 --iters 50000 --bz 100 --nb_trg_labels 400 --DA_FLAG True

# python train_source.py --gpu_num 0 --nb_cnn 6 --bn False --lr 1e-5 --nb_train 100000 --noise 2 --sig_rate 0.035 --bz 400 --optimizer Adam --nb_steps 40000 --clf_v 1
# python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 500000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --trg_clf_param 0 --src_clf_param 1.0 --dataset 'dense'
# 
# parser.add_argument("--gpu", type=int, default = 0)
# parser.add_argument("--docker", type = str2bool, default = True)
# parser.add_argument("--shared", type = str2bool, default = True)
# parser.add_argument("--dis_cnn", type = int, default = 0)
# parser.add_argument("--dis_fc", type=int, default = 128)
# parser.add_argument("--dis_bn", type=str2bool, default = True)
# parser.add_argument("--nD", type = int, default = 1)
# parser.add_argument("--nG", type = int, default = 1)
# parser.add_argument("--acc_up", type = float, default = 0.7)
# parser.add_argument("--acc_down", type = float, default = 0.3)
# parser.add_argument("--lr", type = float, default = 1e-5)
# parser.add_argument("--iters", type = int, default = 1000)
# parser.add_argument("--bz", type = int, default = 400)
# parser.add_argument("--dis_param", type = float, default = 1.0)
# parser.add_argument("--trg_clf_param", type = float, default = 1.0)
# parser.add_argument("--src_clf_param", type = float, default = 1.0)
# parser.add_argument("--source_scratch", type = str2bool, default = True)
# parser.add_argument("--nb_trg_labels", type = int, default = 0)
# parser.add_argument("--fc_layer", type = int, default = 128)
# parser.add_argument("--den_bn", type = str2bool, default = False)
# parser.add_argument("--clf_v", type = int, default = 1)
# parser.add_argument("--dataset", type = str, default = 'dense')

# python TF.py --gpu 0 --docker True --lr 1e-6 --iters 50000 --bz 50 --nb_trg_labels 70 --clf_v 1 --dataset 'total'
# python TF.py --gpu 0 --docker True --lr 1e-6 --iters 20000 --bz 50 --nb_trg_labels 70 --clf_v 1 --dataset 'dense'
# python TF.py --gpu 0 --docker True --lr 1e-6 --iters 20000 --bz 50 --nb_trg_labels 100 --clf_v 1 --dataset 'dense'
# python adda_DA.py --gpu 0 --docker True --dis_cnn 0 --dis_fc 128 --dis_bn True --den_bn False --clf_v 1 --lr 1e-5 --iters 200000 --bz 300 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --nb_trg_labels 0
# python adda_DA.py --gpu 0 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.1 --nb_trg_labels 70 --dataset total
# python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-4 --iters 500000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total
# python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.5 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch False --den_bn False --lr 1e-5 --iters 20000 --bz 400 --nb_trg_labels 1000 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --clf_v 1 --dataset total --valid 100
# python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch False --den_bn False --lr 5e-5 --iters 20000 --bz 400 --nb_trg_labels 1000 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5 --clf_v 1 --dataset total --valid 100

# parser.add_argument("--gpu", type=int, default = 0)
# parser.add_argument("--docker", type = str2bool, default = True)
# parser.add_argument("--shared", type = str2bool, default = True)
# parser.add_argument("--dis_cnn", type = int, default = 0)
# parser.add_argument("--g_cnn", type=int, default = 4)
# parser.add_argument("--g_lr", type=int, default = 1e-5)
# parser.add_argument("--d_lr", type=int, default = 1e-5)
# parser.add_argument("--lsmooth", type=str2bool, default = False)
# parser.add_argument("--dis_fc", type=int, default = 128)
# parser.add_argument("--dis_bn", type=str2bool, default = True)
# parser.add_argument("--nD", type = int, default = 1)
# parser.add_argument("--nG", type = int, default = 1)
# parser.add_argument("--acc_up", type = float, default = 0.8)
# parser.add_argument("--acc_down", type = float, default = 0.3)
# parser.add_argument("--lr", type = float, default = 1e-5)
# parser.add_argument("--iters", type = int, default = 1000)
# parser.add_argument("--bz", type = int, default = 400)
# parser.add_argument("--dis_param", type = float, default = 1.0)
# parser.add_argument("--trg_clf_param", type = float, default = 1.0)
# parser.add_argument("--src_clf_param", type = float, default = 1.0)
# parser.add_argument("--source_scratch", type = str2bool, default = True)
# parser.add_argument("--nb_trg_labels", type = int, default = 0)
# parser.add_argument("--fc_layer", type = int, default = 128)
# parser.add_argument("--den_bn", type = str2bool, default = False)
# parser.add_argument("--clf_v", type = int, default = 1)
# parser.add_argument("--dataset", type = str, default = 'dense')

# python adda_DA3.py --gpu 0 --docker True --dis_cnn 2 --g_cnn 6 --g_lr 1e-5 --d_lr 4e-5 --lsmooth True --dis_bn False --iters 100000 --bz 300 --dis_param 1.0 --trg_clf_param 0 --src_clf_param 3.0 --nb_trg_labels 0 --dataset total

#python adda_DA3.py --gpu 0 --docker True --dis_cnn 2 --g_cnn 4 --g_lr 1e-5 --d_lr 6e-5 --lsmooth False --dis_bn True --den_bn True --iters 400000 --bz 200 --dis_param 1.0 --trg_clf_param 0 --src_clf_param 1.0 --nb_trg_labels 0 --dataset total

python mmd_DA3.py --gpu 0 --docker True --lr 1e-5 --iters 200000 --bz 300 --mmd_param 1.0 --trg_clf_param 1.0 --src_clf_param 1.0 --scratch True --nb_source 100000 --nb_target 85000 --bn True --valid 100
