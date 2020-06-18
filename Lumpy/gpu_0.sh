# python adda_DA.py --gpu 0 --docker True --dis_cnn 0 --dis_fc 256 --shared False --scratch False --dis_bn True
# python mmd_DA.py --gpu 0 --shared True --lr 1e-4 --iters 10000 --bz 300 --scratch False --bn False

python adda_DA.py --gpu 0 --shared True --lr 1e-5 --d_lr 4e-5 --g_lr 1e-5 --iters 200000 --bz 300 --scratch True --nb_source 100000 --nb_target 100000 --dis_bn True --t_blur 3.0