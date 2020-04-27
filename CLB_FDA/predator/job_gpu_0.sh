python mmd_DA.py --gpu 0 --docker True --shared True --lr 1e-5 --iters 100000 --bz 400 --mmd_param 1.0 --nb_trg_labels 0 --source_scratch True
# python train_source.py --gpu_num 0 --nb_cnn 6 --bn True --lr 1e-5 --nb_train 100000 --noise 2 --sig_rate 0.035 --bz 400 --optimizer Adam --nb_steps 10000
