## June 18
# python mmd_wd.py --gpu 2 --docker True --dis_cnn 4 --g_cnn 4 --dis_fc 256 --g_lr 1e-5 --d_lr 1e-5 --lr 1e-5 --dis_bn False --den_bn False --iters 400000 --bz 150 --dis_param 1.0 --trg_clf_param 0 --mmd_param 0 --src_clf_param 1.0 --nb_trg_labels 0 --dataset total

python mmd.py --gpu 2 --g_cnn 6 --fc 512 --c_lr 1e-4 --bz 300 --itr 100000 --d_weight 0.01 --s_weight 1.0
