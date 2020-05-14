## May 14, 2020
# parser.add_argument("--gpu", type=int, default = 2)
# parser.add_argument("--docker", type = str2bool, default = True)
# parser.add_argument("--shared", type = str2bool, default = True)
# parser.add_argument("--lr", type = float, default = 1e-5)
# parser.add_argument("--iters", type = int, default = 10000)
# parser.add_argument("--bz", type = int, default = 300)
# parser.add_argument("--mmd_param", type = float, default = 1.0)
# parser.add_argument("--trg_clf_param", type = float, default = 1.0)
# parser.add_argument("--src_clf_param", type = float, default = 1.0)
# parser.add_argument("--scratch", type = str2bool, default = True)
# parser.add_argument("--nb_source", type = int, default = 100000)
# parser.add_argument("--nb_target", type = int, default = 100000)
# parser.add_argument("--nb_trg_labels", type = int, default = 0)
# parser.add_argument("--fc_layer", type = int, default = 512)
# parser.add_argument("--bn", type = str2bool, default = False)
# parser.add_argument("--s_h", type = float, default = 40)
# parser.add_argument("--s_blur", type = float, default = 0.5)
# parser.add_argument("--s_noise", type = float, default = 10)
# parser.add_argument("--t_h", type = float, default = 50)
# parser.add_argument("--t_blur", type = float, default = 4.0)
# parser.add_argument("--t_noise", type = float, default = 10)
# parser.add_argument("--valid", type = int, default = 100)
# parser.add_argument("--test", type = int, default = 200)

# JOB: python mmd_DA.py --gpu 0 --docker True --shared True --lr 1e-4 --iters 200000 --bz 300 --mmd_param 1 --trg_clf_param 0 --src_clf_param 0.8 --scratch True --nb_source 100000 --nb_target 100000 --nb_trg_labels 0 --fc_layer 128 --bn True --s_h 40 --s_blur 0.5 --s_noise 10 --t_h 50 --t_blur 4 --t_noise 10
# JOB: python mmd_DA.py --gpu 1 --docker True --shared True --lr 1e-4 --iters 200000 --bz 300 --mmd_param 1 --trg_clf_param 0 --src_clf_param 0.8 --scratch True --nb_source 100000 --nb_target 100000 --nb_trg_labels 0 --fc_layer 128 --bn False --s_h 40 --s_blur 0.5 --s_noise 10 --t_h 50 --t_blur 4 --t_noise 10
# JOB: python mmd_DA.py --gpu 2 --docker True --shared True --lr 1e-4 --iters 200000 --bz 300 --mmd_param 1 --trg_clf_param 0 --src_clf_param 0.5 --scratch True --nb_source 100000 --nb_target 100000 --nb_trg_labels 0 --fc_layer 128 --bn False --s_h 40 --s_blur 0.5 --s_noise 10 --t_h 50 --t_blur 4 --t_noise 10
# JOB: python mmd_DA.py --gpu 3 --docker True --shared True --lr 1e-4 --iters 200000 --bz 300 --mmd_param 1 --trg_clf_param 0 --src_clf_param 0.2 --scratch True --nb_source 100000 --nb_target 100000 --nb_trg_labels 0 --fc_layer 128 --bn False --s_h 40 --s_blur 0.5 --s_noise 10 --t_h 50 --t_blur 4 --t_noise 10

JOB: python mmd_DA.py --gpu 0 --docker True --shared True --lr 1e-4 --iters 200000 --bz 300 --mmd_param 0.8 --trg_clf_param 0 --src_clf_param 1.0 --scratch True --nb_source 100000 --nb_target 100000 --nb_trg_labels 0 --fc_layer 128 --bn True --s_h 40 --s_blur 0.5 --s_noise 10 --t_h 50 --t_blur 4 --t_noise 10
JOB: python mmd_DA.py --gpu 1 --docker True --shared True --lr 1e-4 --iters 200000 --bz 300 --mmd_param 0.8 --trg_clf_param 0 --src_clf_param 1.0 --scratch True --nb_source 100000 --nb_target 100000 --nb_trg_labels 0 --fc_layer 128 --bn False --s_h 40 --s_blur 0.5 --s_noise 10 --t_h 50 --t_blur 4 --t_noise 10
JOB: python mmd_DA.py --gpu 2 --docker True --shared True --lr 1e-4 --iters 200000 --bz 300 --mmd_param 0.5 --trg_clf_param 0 --src_clf_param 1.0 --scratch True --nb_source 100000 --nb_target 100000 --nb_trg_labels 0 --fc_layer 128 --bn False --s_h 40 --s_blur 0.5 --s_noise 10 --t_h 50 --t_blur 4 --t_noise 10
JOB: python mmd_DA.py --gpu 3 --docker True --shared True --lr 1e-4 --iters 200000 --bz 300 --mmd_param 0.2 --trg_clf_param 0 --src_clf_param 1.0 --scratch True --nb_source 100000 --nb_target 100000 --nb_trg_labels 0 --fc_layer 128 --bn False --s_h 40 --s_blur 0.5 --s_noise 10 --t_h 50 --t_blur 4 --t_noise 10







