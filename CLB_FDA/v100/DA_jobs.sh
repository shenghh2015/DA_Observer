# ### Apr. 22, 2020
# python train_DA.py --gpu 0 --dis_cnn 4 --dis_fc 256 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.8 --dAcc2 0.9 --iters 200000 --bz 400
# python train_DA.py --gpu 1 --dis_cnn 4 --dis_fc 256 --dis_bn True --D_lr 1e-4 --G_lr 1e-4 --nD 1 --nG 1 --dAcc1 0.8 --dAcc2 0.9 --iters 200000 --bz 400
# python train_DA.py --gpu 2 --dis_cnn 4 --dis_fc 256 --dis_bn True --D_lr 5e-5 --G_lr 5e-5 --nD 1 --nG 1 --dAcc1 0.8 --dAcc2 0.9 --iters 200000 --bz 400
# python train_DA.py --gpu 3 --dis_cnn 4 --dis_fc 256 --dis_bn True --D_lr 5e-4 --G_lr 5e-4 --nD 1 --nG 1 --dAcc1 0.8 --dAcc2 0.9 --iters 200000 --bz 400
# 
# python train_DA.py --gpu 4 --dis_cnn 4 --dis_fc 256 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.85 --dAcc2 0.95 --iters 200000 --bz 400
# python train_DA.py --gpu 5 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-6 --nD 1 --nG 1 --dAcc1 0.85 --dAcc2 0.95 --iters 200000 --bz 400

# JOB: python train_DA.py --gpu 0 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-7 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 1000
# JOB: python train_DA.py --gpu 1 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-6 --G_lr 1e-7 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 1000
# JOB: python train_DA.py --gpu 2 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-7 --G_lr 1e-7 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 1000
# JOB: python train_DA.py --gpu 3 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-7 --G_lr 1e-8 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 1000

## Apr. 23, 2020
#JOB: python train_DA1.py --gpu 0 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 10 --bz 400 --lamda 0.1

#JOB: python train_DA1.py --gpu 0 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 0.1
#JOB: python train_DA1.py --gpu 1 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 0.01
#JOB: python train_DA1.py --gpu 2 --dis_cnn 4 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 0.001
#JOB: python train_DA1.py --gpu 3 --dis_cnn 0 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 1.0

#JOB: python train_DA1.py --gpu 0 --dis_cnn 0 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 0.1
#JOB: python train_DA1.py --gpu 1 --dis_cnn 0 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 0.01
#JOB: python train_DA1.py --gpu 2 --dis_cnn 0 --dis_fc 128 --dis_bn True --D_lr 1e-5 --G_lr 1e-5 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 0.001
#JOB: python train_DA1.py --gpu 3 --dis_cnn 0 --dis_fc 128 --dis_bn True --D_lr 1e-6 --G_lr 1e-6 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 1.0

# JOB: python train_DA1.py --gpu 0 --dis_cnn 4 --dis_fc 64 --dis_bn True --D_lr 1e-6 --G_lr 1e-6 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 10.0
# JOB: python train_DA1.py --gpu 1 --dis_cnn 4 --dis_fc 64 --dis_bn True --D_lr 1e-6 --G_lr 1e-6 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 100.0
# JOB: python train_DA1.py --gpu 2 --dis_cnn 4 --dis_fc 64 --dis_bn True --D_lr 1e-6 --G_lr 1e-6 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 1.0
# JOB: python train_DA1.py --gpu 3 --dis_cnn 4 --dis_fc 64 --dis_bn True --D_lr 1e-6 --G_lr 1e-6 --nD 1 --nG 1 --dAcc1 0.80 --dAcc2 0.90 --iters 100000 --bz 400 --lamda 1.0

## Apr. 26, 2020
# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --lr 1e-3 --iters 100000 --bz 400 --mmd_param 1.0 --nb_trg_labels 0
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --lr 1e-4 --iters 100000 --bz 400 --mmd_param 1.0 --nb_trg_labels 0
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --lr 1e-5 --iters 100000 --bz 400 --mmd_param 1.0 --nb_trg_labels 0
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --lr 1e-6 --iters 100000 --bz 400 --mmd_param 1.0 --nb_trg_labels 0

# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --lr 1e-5 --iters 100000 --bz 400 --mmd_param 0.5 --nb_trg_labels 0
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --lr 1e-4 --iters 100000 --bz 400 --mmd_param 0.5 --nb_trg_labels 0
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --lr 1e-5 --iters 100000 --bz 400 --mmd_param 2.0 --nb_trg_labels 0
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --lr 1e-4 --iters 100000 --bz 400 --mmd_param 2.0 --nb_trg_labels 0

## Apr. 27, 2020
# DA + target labels
# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 100000 --bz 400 --nb_trg_labels 200 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 100000 --bz 400 --nb_trg_labels 300 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 100000 --bz 400 --nb_trg_labels 400 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch True --den_bn False --lr 1e-5 --iters 100000 --bz 400 --nb_trg_labels 500 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0

# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-3 --iters 100000 --bz 400 --nb_trg_labels 200 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch True --den_bn False --lr 1e-3 --iters 100000 --bz 400 --nb_trg_labels 300 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 1e-3 --iters 100000 --bz 400 --nb_trg_labels 400 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch True --den_bn False --lr 1e-3 --iters 100000 --bz 400 --nb_trg_labels 500 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0

JOB: python mmd_DA.py --gpu 0 --docker True --shared True --source_scratch True --den_bn False --lr 1e-4 --iters 100000 --bz 400 --nb_trg_labels 200 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
JOB: python mmd_DA.py --gpu 1 --docker True --shared True --source_scratch True --den_bn False --lr 1e-4 --iters 100000 --bz 400 --nb_trg_labels 300 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
JOB: python mmd_DA.py --gpu 2 --docker True --shared True --source_scratch True --den_bn False --lr 1e-4 --iters 100000 --bz 400 --nb_trg_labels 400 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0
JOB: python mmd_DA.py --gpu 3 --docker True --shared True --source_scratch True --den_bn False --lr 1e-4 --iters 100000 --bz 400 --nb_trg_labels 500 --mmd_param 1.0 --src_clf_param 1.0 --trg_clf_param 1.0


# JOB: python TF.py --gpu 0 --docker True --lr 1e-6 --iters 100000 --bz 100 --nb_trg_labels 100
# JOB: python TF.py --gpu 1 --docker True --lr 1e-6 --iters 100000 --bz 100 --nb_trg_labels 200
# JOB: python TF.py --gpu 2 --docker True --lr 1e-6 --iters 100000 --bz 100 --nb_trg_labels 300
# JOB: python TF.py --gpu 3 --docker True --lr 1e-6 --iters 100000 --bz 100 --nb_trg_labels 400

# JOB: python TF.py --gpu 0 --docker True --lr 1e-5 --iters 100000 --bz 100 --nb_trg_labels 100
# JOB: python TF.py --gpu 1 --docker True --lr 1e-5 --iters 100000 --bz 100 --nb_trg_labels 200
# JOB: python TF.py --gpu 2 --docker True --lr 1e-5 --iters 100000 --bz 100 --nb_trg_labels 300
# JOB: python TF.py --gpu 3 --docker True --lr 1e-5 --iters 100000 --bz 100 --nb_trg_labels 400

