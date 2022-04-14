#!bin/bash
python train.py --seed 2 --aug_size 20000 --lr 1e-5 --data_for_train both --path 22layer_8_false_saved_models_schedule_1e-5_both_95_aug2w
python train.py --seed 2 --aug_size 20000 --lr 5e-6 --data_for_train both --path 22layer_8_false_saved_models_schedule_5e-6_both_95_aug2w
python train.py --seed 2 --aug_size 30000 --lr 1e-5 --data_for_train both --path 22layer_8_false_saved_models_schedule_1e-5_both_95_aug3w
python train.py --seed 2 --aug_size 30000 --lr 5e-6 --data_for_train both --path 22layer_8_false_saved_models_schedule_5e-6_both_95_aug3w
# python train.py --seed 1 --lr 1e-5 --data_for_train both --path layer_8_false_saved_models_schedule_1e-5_1_both_80_aug1w
# python train.py --seed 1 --lr 1e-5 --data_for_train ar --path layer_8_false_saved_models_schedule_1e-5_1_ar_80_aug1w
# python train.py --seed 1 --lr 1e-5 --data_for_train gk --path layer_8_false_saved_models_schedule_1e-5_1_gk_80_aug1w
# python train.py --seed 1 --lr 5e-6 --data_for_train both --path layer_8_false_saved_models_schedule_5e-6_1_both_80_aug1w
# python train.py --seed 1 --lr 5e-6 --data_for_train ar --path layer_8_false_saved_models_schedule_5e-6_1_ar_80_aug1w
# python train.py --seed 1 --lr 5e-6 --data_for_train gk --path layer_8_false_saved_models_schedule_5e-6_1_gk_80_aug1w
# python train.py --seed 0 --lr 1e-7 --data_for_train ar --path false_saved_models_1e-7_0_ar
# python train.py --seed 1 --lr 1e-7 --data_for_train ar --path false_saved_models_1e-7_1_ar
# python train.py --seed 2 --lr 1e-7 --data_for_train ar --path false_saved_models_1e-7_2_ar
# python train.py --seed 0 --lr 5e-6 --data_for_train ar --path false_saved_models_5e-6_0_ar
# python train.py --seed 1 --lr 5e-6 --data_for_train ar --path false_saved_models_5e-6_1_ar
# python train.py --seed 2 --lr 5e-6 --data_for_train ar --path false_saved_models_5e-6_2_ar
# python train.py --seed 0 --lr 1e-6 --data_for_train ar --path false_saved_models_1e-6_0_ar
# python train.py --seed 1 --lr 1e-6 --data_for_train ar --path false_saved_models_1e-6_1_ar
# python train.py --seed 2 --lr 1e-6 --data_for_train ar --path false_saved_models_1e-6_2_ar
# python train.py --seed 0 --lr 1e-7 --data_for_train gk --path saved_models_1e-7_0_gk
# python train.py --seed 1 --lr 1e-7 --data_for_train gk --path saved_models_1e-7_1_gk
# python train.py --seed 2 --lr 1e-7 --data_for_train gk --path saved_models_1e-7_2_gk
# python train.py --seed 0 --lr 5e-6 --data_for_train gk --path saved_models_5e-6_0_gk
# python train.py --seed 1 --lr 5e-6 --data_for_train gk --path saved_models_5e-6_1_gk
# python train.py --seed 2 --lr 5e-6 --data_for_train gk --path saved_models_5e-6_2_gk
# python train.py --seed 0 --lr 1e-6 --data_for_train gk --path saved_models_1e-6_0_gk
# python train.py --seed 1 --lr 1e-6 --data_for_train gk --path saved_models_1e-6_1_gk
# python train.py --seed 2 --lr 1e-6 --data_for_train gk --path saved_models_1e-6_2_gk