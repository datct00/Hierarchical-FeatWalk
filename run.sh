gpuid=0

python eval.py --gpu ${gpuid}  --n_episodes 2000  --n_aug_support_samples 17 --n_shot 1 --distill_model /mnt/HDD2/lamle/FeatWalk/FeatWalk/ResNet12_stl_deepbdc_distill/last_model.tar --test_times 1  --lr 0.5 --fix_seed --sfc_bs 3 --sim_temperature 32
python eval.py --gpu ${gpuid}  --n_episodes 2000  --n_aug_support_samples 17 --n_shot 5 --distill_model /mnt/HDD2/lamle/FeatWalk/FeatWalk/ResNet12_stl_deepbdc_distill/last_model.tar --test_times 1  --lr 0.01 --fix_seed --sfc_bs 3 --sim_temperature 32