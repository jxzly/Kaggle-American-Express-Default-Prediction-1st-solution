python S1_denoise.py
python S2_manual_feature.py
python S3_series_feature.py
python S4_feature_combined.py
python S5_LGB_main.py
CUDA_VISIBLE_DEVICES=0 python  S6_NN_main.py --do_train --batch_size 512
python S7_ensemble.py
