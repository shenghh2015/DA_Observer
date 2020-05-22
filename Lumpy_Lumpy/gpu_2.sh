# python train_source.py --gpu_num 2 --nb_cnn 4 --bn False --lr 5e-5 --nb_train 100000 --noise 2.0 --sig_rate 0.035 --bz 400 --optimizer 'Adam' --nb_steps 100000
# parser.add_argument("--gpu", type=int)
# parser.add_argument("--docker", type = str2bool, default = True)
# # parser.add_argument("--shared", type = str2bool, default = True)
# parser.add_argument("--lr", type = float)
# parser.add_argument("--iters", type = int)
# parser.add_argument("--bz", type = int)
# parser.add_argument("--source_scratch", type = str2bool, default = False)
# parser.add_argument("--nb_trg_labels", type = int, default = 0)
# parser.add_argument("--fc_layer", type = int, default = 128)

# python TF.py --gpu 2 --docker True --lr 1e-5 --iters 10000 --bz 100 --source_scratch True --nb_trg_labels 400 --fc_layer 128
# python TF.py --gpu 2 --docker True --lr 1e-5 --iters 10000 --bz 100 --source_scratch False --nb_trg_labels 500 --fc_layer 128
# python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 500 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.5
#python train_source.py --gpu_num 2 --nb_cnn 4 --bn False --lr 1e-5 --nb_train 100000 --noise 2 --sig_rate 0.035 --bz 400 --optimizer Adam --nb_steps 50000
# python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --clf_v 2
# python train_source.py --gpu_num 2 --nb_cnn 6 --bn False --lr 1e-5 --nb_train 100000 --noise 2 --sig_rate 0.035 --bz 400 --optimizer Adam --nb_steps 100000 --clf_v 1

# python TF.py --gpu 2 --docker True --lr 1e-6 --iters 50000 --bz 100 --nb_trg_labels 200 --clf_v 1 --dataset 'total'
# python TF.py --gpu 2 --docker True --lr 1e-6 --iters 20000 --bz 100 --nb_trg_labels 400 --clf_v 1 --dataset 'dense'
# python adda_DA.py --gpu 2 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.05 --nb_trg_labels 70 --dataset dense
# python adda_DA.py --gpu 2 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch False --den_bn False --clf_v 1 --lr 1e-5 --iters 200000 --bz 300 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --nb_trg_labels 100
# python adda_DA.py --gpu 2 --docker True --dis_cnn 4 --dis_fc 128 --dis_bn True --source_scratch True --den_bn False --clf_v 1 --lr 1e-5 --iters 100000 --bz 400 --dis_param 1.0 --src_clf_param 1.0 --trg_clf_param 0.0 --nb_trg_labels 0 --dataset total
# python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 500000 --bz 400 --nb_trg_labels 0 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total
# python total_val_100.py 2 true
# python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 5e-5 --iters 50000 --bz 400 --nb_trg_labels 0 --mmd_param 0.5 --src_clf_param 1.0 --trg_clf_param 0 --clf_v 1 --dataset total --valid 100
# python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch False --den_bn False --lr 5e-5 --iters 20000 --bz 400 --nb_trg_labels 1000 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --clf_v 1 --dataset total --valid 100
# python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch False --den_bn False --lr 1e-4 --iters 20000 --bz 400 --nb_trg_labels 1000 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0 --clf_v 1 --dataset total --valid 100

# python adda_DA2.py --gpu 2 --docker True --dis_cnn 2 --g_cnn 6 --g_lr 1e-5 --d_lr 4e-5 --lsmooth True --dis_bn False --iters 100000 --bz 300 --dis_param 1.0 --trg_clf_param 0 --src_clf_param 5.0 --nb_trg_labels 0 --dataset total
python adda_DA.py --gpu 2 --shared True --lr 1e-5 --d_lr 1e-5 --g_lr 1e-5 --iters 200000 --bz 300 --scratch True --nb_source 100000 --nb_target 100000 --dis_bn True --t_blur 3.0