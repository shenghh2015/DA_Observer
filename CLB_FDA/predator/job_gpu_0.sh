# python train_source.py --gpu_num 0 --nb_cnn 6 --bn True --lr 1e-5 --nb_train 100000 --noise 2 --sig_rate 0.035 --bz 400 --optimizer Adam --nb_steps 10000
# python mmd_DA.py --gpu 0 --docker True --shared True --lr 1e-5 --iters 100000 --bz 400 --mmd_param 0.5 --nb_trg_labels 0
# python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 100000 --bz 400 --nb_trg_labels 500 --mmd_param 1.0 --trg_clf_param 1.0 --src_clf_param 1.0
python TF.py --gpu 0 --docker True --lr 1e-6 --iters 50000 --bz 100 --nb_trg_labels 400 --DA_FLAG True
