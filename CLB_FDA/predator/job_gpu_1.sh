# python coral_DA.py --gpu 1 --docker True --shared True --lr 1e-5 --iters 200000 --bz 300 --mmd_param 10.0 --nb_trg_labels 0 --source_scratch True
# python train_source.py --gpu_num 1 --nb_cnn 6 --bn True --lr 1e-5 --nb_train 100000 --noise 2 --sig_rate 0.035 --bz 400 --optimizer Adam --nb_steps 10000
#python mmd_DA.py --gpu 1 --docker True --shared True --lr 1e-5 --iters 100000 --bz 400 --mmd_param 0.0 --nb_trg_labels 100 --source_scratch False --src_clf_param 0.0 --trg_clf_param 1.0
python source_source.py
