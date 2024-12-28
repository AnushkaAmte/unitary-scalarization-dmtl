cd ..

python3 supervised_experiments/test_chexphoto.py --net_basename Pne_Nor_Cov_baseline-lr:0.001-wd:0.0_sleek-wave-341 --model_type last --task_labels Pne_Nor_Cov --data_labels Pne_Nor_Cov

python3 supervised_experiments/test_chexphoto.py --net_basename 0_1_7_14_r:1_baseline-lr:0.001-wd:0.0_clear-snow-75 --model_type last --task_labels 0_1_7_14 --data_labels 0_1_7_14
printf "Done with Inf_Pne_Nor_Cov\n"

python3 supervised_experiments/test_chexphoto.py --net_basename 0_2_7_14_r:1_baseline-lr:0.001-wd:0.0_zany-dream-80 --model_type last --task_labels 0_2_7_14 --data_labels 0_2_7_14 
printf "Done with Eff_Pne_Nor_Cov\n"

python3 supervised_experiments/test_chexphoto.py --net_basename 0_3_7_14_r:1_baseline-lr:0.001-wd:0.0_worthy-bush-86 --model_type last --task_labels 0_3_7_14 --data_labels 0_3_7_14
printf "Done with Ate_Pne_Nor_Cov\n"

python3 supervised_experiments/test_chexphoto.py --net_basename 0_4_7_14_r:1_baseline-lr:0.001-wd:0.0_gentle-grass-107 --model_type last --task_labels 0_4_7_14 --data_labels 0_4_7_14
printf "Done with Mas_Pne_Nor_Cov\n"

python3 supervised_experiments/test_chexphoto.py --net_basename 0_5_7_14_r:1_baseline-lr:0.001-wd:0.0_sunny-thunder-72 --model_type last --task_labels 0_5_7_14 --data_labels 0_5_7_14
printf "Done with Pne_Ple_Nor_Cov\n"

python3 supervised_experiments/test_chexphoto.py --net_basename 0_6_7_14_r:1_baseline-lr:0.001-wd:0.0_zany-oath-81 --model_type last --task_labels 0_6_7_14 --data_labels 0_6_7_14
printf "Done with Nod_Pne_Nor_Cov\n"

python3 supervised_experiments/test_chexphoto.py --net_basename 0_7_8_14_r:1_baseline-lr:0.001-wd:0.0_dulcet-microwave-91 --model_type last --task_labels 0_7_8_14 --data_labels 0_7_8_14
printf "Done with Car_Pne_Nor_Cov\n"

python3 supervised_experiments/test_chexphoto.py --net_basename 0_7_9_14_r:1_baseline-lr:0.001-wd:0.0_worldly-moon-94 --model_type last --task_labels 0_7_9_14 --data_labels 0_7_9_14
printf "Done with Con_Pne_Nor_Cov\n"

python3 supervised_experiments/test_chexphoto.py --net_basename 0_7_10_14_r:1_baseline-lr:0.001-wd:0.0_effortless-fire-83 --model_type last --task_labels 0_7_10_14 --data_labels 0_7_10_14
printf "Done with Ede_Pne_Nor_Cov\n"

python3 supervised_experiments/test_chexphoto.py --net_basename 0_7_11_14_r:1_baseline-lr:0.001-wd:0.0_swift-disco-118 --model_type last --task_labels 0_7_11_14 --data_labels 0_7_11_14
printf "Done with Emp_Pne_Nor_Cov\n"

python3 supervised_experiments/test_chexphoto.py --net_basename 0_7_12_14_r:1_baseline-lr:0.001-wd:0.0_clean-dew-131 --model_type last --task_labels 0_7_12_14 --data_labels 0_7_12_14
printf "Done with Fib_Pne_Nor_Cov\n"

python3 supervised_experiments/test_chexphoto.py --net_basename 0_7_13_14_r:1_baseline-lr:0.001-wd:0.0_still-galaxy-142 --model_type last --task_labels 0_7_13_14 --data_labels 0_7_13_14
printf "Done with Her_Pne_Nor_Cov\n"

python3 supervised_experiments/test_chexphoto.py --net_basename Pnt_Pne_Nor_Cov_r:1_baseline-lr:0.001-wd:0.0_usual-fog-567 --model_type last --task_labels Pnt_Pne_Nor_Cov --data_labels Pnt_Pne_Nor_Cov
printf "Done with Pnt_Pne_Nor_Cov\n"

cp  0_1_7_14_r:1_baseline-lr:0.001-wd:0.0_clear-snow-75_last_model.pkl \
0_2_7_14_r:1_baseline-lr:0.001-wd:0.0_zany-dream-80_last_model.pkl \
0_3_7_14_r:1_baseline-lr:0.001-wd:0.0_worthy-bush-86_last_model.pkl\
0_4_7_14_r:1_baseline-lr:0.001-wd:0.0_gentle-grass-107_last_model.pkl \
0_5_7_14_r:1_baseline-lr:0.001-wd:0.0_sunny-thunder-72_last_model.pkl\
0_6_7_14_r:1_baseline-lr:0.001-wd:0.0_zany-oath-81_last_model.pkl \
0_7_8_14_r:1_baseline-lr:0.001-wd:0.0_dulcet-microwave-91_last_model.pkl \
0_7_9_14_r:1_baseline-lr:0.001-wd:0.0_worldly-moon-94_last_model.pkl \
0_7_10_14_r:1_baseline-lr:0.001-wd:0.0_effortless-fire-83_last_model.pkl \
0_7_11_14_r:1_baseline-lr:0.001-wd:0.0_swift-disco-118_last_model.pkl \
0_7_12_14_r:1_baseline-lr:0.001-wd:0.0_clean-dew-131_last_model.pkl \
0_7_13_14_r:1_baseline-lr:0.001-wd:0.0_still-galaxy-142_last_model.pkl \
/data6/anushkapa_scratch/unitary-scalarization-dmtl/chexphoto/saved_models


python3 supervised_experiments/heatmap.py --net_basename 0_1_7_14_r:1_baseline-lr:0.001-wd:0.0_clear-snow-75 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq --nih_labels Nor_EnC_Pne_Cov --method grad-norm-sq
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq/0_1_7_14_r:1_baseline-lr:0.001-wd:0.0_clear-snow-75 --model_cam_name 0_1_7_14_r:1_baseline-lr:0.001-wd:0.0_clear-snow-75

python3 supervised_experiments/heatmap.py --net_basename 0_2_7_14_r:1_baseline-lr:0.001-wd:0.0_zany-dream-80 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq --nih_labels Nor_Car_Pne_Cov --method grad-norm-sq
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq/0_2_7_14_r:1_baseline-lr:0.001-wd:0.0_zany-dream-80 --model_cam_name 0_2_7_14_r:1_baseline-lr:0.001-wd:0.0_zany-dream-80

python3 supervised_experiments/heatmap.py --net_basename 0_3_7_14_r:1_baseline-lr:0.001-wd:0.0_worthy-bush-86 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq --nih_labels Nor_LunO_Pne_Cov --method grad-norm-sq
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq/0_3_7_14_r:1_baseline-lr:0.001-wd:0.0_worthy-bush-86 --model_cam_name 0_3_7_14_r:1_baseline-lr:0.001-wd:0.0_worthy-bush-86

python3 supervised_experiments/heatmap.py --net_basename 0_4_7_14_r:1_baseline-lr:0.001-wd:0.0_gentle-grass-107 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq --nih_labels Nor_LunL_Pne_Cov --method grad-norm-sq
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq/0_4_7_14_r:1_baseline-lr:0.001-wd:0.0_gentle-grass-107 --model_cam_name 0_4_7_14_r:1_baseline-lr:0.001-wd:0.0_gentle-grass-107

python3 supervised_experiments/heatmap.py --net_basename 0_5_7_14_r:1_baseline-lr:0.001-wd:0.0_sunny-thunder-72 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq --nih_labels Nor_Ede_Pne_Cov --method grad-norm-sq
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq/0_5_7_14_r:1_baseline-lr:0.001-wd:0.0_sunny-thunder-72 --model_cam_name 0_5_7_14_r:1_baseline-lr:0.001-wd:0.0_sunny-thunder-72

python3 supervised_experiments/heatmap.py --net_basename 0_6_7_14_r:1_baseline-lr:0.001-wd:0.0_zany-oath-81 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq --nih_labels Nor_Con_Pne_Cov --method grad-norm-sq
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq/0_6_7_14_r:1_baseline-lr:0.001-wd:0.0_zany-oath-81 --model_cam_name 0_6_7_14_r:1_baseline-lr:0.001-wd:0.0_zany-oath-81

python3 supervised_experiments/heatmap.py --net_basename 0_7_8_14_r:1_baseline-lr:0.001-wd:0.0_dulcet-microwave-91 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq --nih_labels Nor_Pne_Ate_Cov --method grad-norm-sq
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq/0_7_8_14_r:1_baseline-lr:0.001-wd:0.0_dulcet-microwave-91 --model_cam_name 0_7_8_14_r:1_baseline-lr:0.001-wd:0.0_dulcet-microwave-91

python3 supervised_experiments/heatmap.py --net_basename 0_7_9_14_r:1_baseline-lr:0.001-wd:0.0_worldly-moon-94 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq --nih_labels Nor_Pne_Pnt_Cov --method grad-norm-sq
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq/0_7_9_14_r:1_baseline-lr:0.001-wd:0.0_worldly-moon-94 --model_cam_name 0_7_9_14_r:1_baseline-lr:0.001-wd:0.0_worldly-moon-94

python3 supervised_experiments/heatmap.py --net_basename 0_7_10_14_r:1_baseline-lr:0.001-wd:0.0_effortless-fire-83 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq --nih_labels Nor_Pne_Ple_Cov --method grad-norm-sq
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq/0_7_10_14_r:1_baseline-lr:0.001-wd:0.0_effortless-fire-83 --model_cam_name 0_7_10_14_r:1_baseline-lr:0.001-wd:0.0_effortless-fire-83

python3 supervised_experiments/heatmap.py --net_basename 0_7_11_14_r:1_baseline-lr:0.001-wd:0.0_swift-disco-118 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq --nih_labels Nor_Pne_PleO_Cov --method grad-norm-sq
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq/0_7_11_14_r:1_baseline-lr:0.001-wd:0.0_swift-disco-118 --model_cam_name 0_7_11_14_r:1_baseline-lr:0.001-wd:0.0_swift-disco-118

python3 supervised_experiments/heatmap.py --net_basename 0_7_12_14_r:1_baseline-lr:0.001-wd:0.0_clean-dew-131 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq --nih_labels Nor_Pne_Fra_Cov --method grad-norm-sq
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq/0_7_12_14_r:1_baseline-lr:0.001-wd:0.0_clean-dew-131 --model_cam_name 0_7_12_14_r:1_baseline-lr:0.001-wd:0.0_clean-dew-131

python3 supervised_experiments/heatmap.py --net_basename 0_7_13_14_r:1_baseline-lr:0.001-wd:0.0_still-galaxy-142 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq --nih_labels Nor_Pne_Sup_Cov --method grad-norm-sq
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_norm_sq/0_7_13_14_r:1_baseline-lr:0.001-wd:0.0_still-galaxy-142 --model_cam_name 0_7_13_14_r:1_baseline-lr:0.001-wd:0.0_still-galaxy-142








python3 supervised_experiments/heatmap.py --net_basename 0_1_7_14_r:1_baseline-lr:0.001-wd:0.0_clear-snow-75 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam --nih_labels Nor_EnC_Pne_Cov --method grad-cam
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam/0_1_7_14_r:1_baseline-lr:0.001-wd:0.0_clear-snow-75 --model_cam_name 0_1_7_14_r:1_baseline-lr:0.001-wd:0.0_clear-snow-75

CUDA_VISIBLE_DEVICES=7 python3 supervised_experiments/heatmap.py --net_basename 0_2_7_14_r:1_baseline-lr:0.001-wd:0.0_zany-dream-80 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam --nih_labels Nor_Car_Pne_Cov --method grad-cam
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam/0_2_7_14_r:1_baseline-lr:0.001-wd:0.0_zany-dream-80 --model_cam_name 0_2_7_14_r:1_baseline-lr:0.001-wd:0.0_zany-dream-80

CUDA_VISIBLE_DEVICES=0  python3 supervised_experiments/heatmap.py --net_basename 0_3_7_14_r:1_baseline-lr:0.001-wd:0.0_worthy-bush-86 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam --nih_labels Nor_LunO_Pne_Cov --method grad-cam
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam/0_3_7_14_r:1_baseline-lr:0.001-wd:0.0_worthy-bush-86 --model_cam_name 0_3_7_14_r:1_baseline-lr:0.001-wd:0.0_worthy-bush-86

CUDA_VISIBLE_DEVICES=0  python3 supervised_experiments/heatmap.py --net_basename 0_4_7_14_r:1_baseline-lr:0.001-wd:0.0_gentle-grass-107 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam --nih_labels Nor_LunL_Pne_Cov --method grad-cam
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam/0_4_7_14_r:1_baseline-lr:0.001-wd:0.0_gentle-grass-107 --model_cam_name 0_4_7_14_r:1_baseline-lr:0.001-wd:0.0_gentle-grass-107

CUDA_VISIBLE_DEVICES=0  python3 supervised_experiments/heatmap.py --net_basename 0_5_7_14_r:1_baseline-lr:0.001-wd:0.0_sunny-thunder-72 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam --nih_labels Nor_Ede_Pne_Cov --method grad-cam
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam/0_5_7_14_r:1_baseline-lr:0.001-wd:0.0_sunny-thunder-72 --model_cam_name 0_5_7_14_r:1_baseline-lr:0.001-wd:0.0_sunny-thunder-72

CUDA_VISIBLE_DEVICES=4 supervised_experiments/heatmap.py --net_basename 0_6_7_14_r:1_baseline-lr:0.001-wd:0.0_zany-oath-81 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam --nih_labels Nor_Con_Pne_Cov --method grad-cam
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam/0_6_7_14_r:1_baseline-lr:0.001-wd:0.0_zany-oath-81 --model_cam_name 0_6_7_14_r:1_baseline-lr:0.001-wd:0.0_zany-oath-81

CUDA_VISBILE_DEVICES=7 python3 supervised_experiments/heatmap.py --net_basename 0_7_8_14_r:1_baseline-lr:0.001-wd:0.0_dulcet-microwave-91 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam --nih_labels Nor_Pne_Ate_Cov --method grad-cam
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam/0_7_8_14_r:1_baseline-lr:0.001-wd:0.0_dulcet-microwave-91 --model_cam_name 0_7_8_14_r:1_baseline-lr:0.001-wd:0.0_dulcet-microwave-91

python3 supervised_experiments/heatmap.py --net_basename 0_7_9_14_r:1_baseline-lr:0.001-wd:0.0_worldly-moon-94 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam --nih_labels Nor_Pne_Pnt_Cov --method grad-cam
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam/0_7_9_14_r:1_baseline-lr:0.001-wd:0.0_worldly-moon-94 --model_cam_name 0_7_9_14_r:1_baseline-lr:0.001-wd:0.0_worldly-moon-94

python3 supervised_experiments/heatmap.py --net_basename 0_7_10_14_r:1_baseline-lr:0.001-wd:0.0_effortless-fire-83 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam --nih_labels Nor_Pne_Ple_Cov --method grad-cam
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam/0_7_10_14_r:1_baseline-lr:0.001-wd:0.0_effortless-fire-83 --model_cam_name 0_7_10_14_r:1_baseline-lr:0.001-wd:0.0_effortless-fire-83

python3 supervised_experiments/heatmap.py --net_basename 0_7_11_14_r:1_baseline-lr:0.001-wd:0.0_swift-disco-118 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam --nih_labels Nor_Pne_PleO_Cov --method grad-cam
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam/0_7_11_14_r:1_baseline-lr:0.001-wd:0.0_swift-disco-118 --model_cam_name 0_7_11_14_r:1_baseline-lr:0.001-wd:0.0_swift-disco-118

python3 supervised_experiments/heatmap.py --net_basename 0_7_12_14_r:1_baseline-lr:0.001-wd:0.0_clean-dew-131 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam --nih_labels Nor_Pne_Fra_Cov --method grad-cam
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam/0_7_12_14_r:1_baseline-lr:0.001-wd:0.0_clean-dew-131 --model_cam_name 0_7_12_14_r:1_baseline-lr:0.001-wd:0.0_clean-dew-131

python3 supervised_experiments/heatmap.py --net_basename 0_7_13_14_r:1_baseline-lr:0.001-wd:0.0_still-galaxy-142 --model_type last --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam --nih_labels Nor_Pne_Sup_Cov --method grad-cam
python3 supervised_experiments/ols_score.py --heatmap_dir /data6/anushkapa_scratch/unitary-scalarization-dmtl/heatmaps/grad_cam/0_7_13_14_r:1_baseline-lr:0.001-wd:0.0_still-galaxy-142 --model_cam_name 0_7_13_14_r:1_baseline-lr:0.001-wd:0.0_still-galaxy-142







python3 supervised_experiments/test_vindr.py --net_basename vindr_0-lr:0.01-wd:0.0005_spring-donkey-30 --model_name last

python3 supervised_experiments/test_vindr.py --net_basename vindr_1-lr:0.01-wd:0.0005_dandy-durian-17 --model_name last

python3 supervised_experiments/test_vindr.py --net_basename vindr_2-lr:0.01-wd:0.0005_solar-violet-18 --model_name last

python3 supervised_experiments/test_vindr.py --net_basename vindr_3-lr:0.01-wd:0.0005_silver-eon-19 --model_name last

python3 supervised_experiments/test_vindr.py --net_basename vindr_4-lr:0.01-wd:0.0005_quiet-sun-20 --model_name last

python3 supervised_experiments/test_vindr.py --net_basename vindr_5-lr:0.01-wd:0.0005_radiant-field-21 --model_name last

python3 supervised_experiments/test_vindr.py --net_basename vindr_6-lr:0.01-wd:0.0005_silver-river-22 --model_name last

python3 supervised_experiments/test_vindr.py --net_basename vindr_7-lr:0.01-wd:0.0005_bumbling-elevator-23 --model_name last

python3 supervised_experiments/test_vindr.py --net_basename vindr_8-lr:0.01-wd:0.0005_fresh-capybara-24 --model_name last

python3 supervised_experiments/test_vindr.py --net_basename vindr_9-lr:0.01-wd:0.0005_feasible-mountain-25 --model_name last

python3 supervised_experiments/test_vindr.py --net_basename vindr_10-lr:0.01-wd:0.0005_unique-field-26 --model_name last

python3 supervised_experiments/test_vindr.py --net_basename vindr_11-lr:0.01-wd:0.0005_stellar-morning-27 --model_name last

python3 supervised_experiments/test_vindr.py --net_basename vindr_12-lr:0.01-wd:0.0005_dulcet-resonance-28 --model_name last

python3 supervised_experiments/test_vindr.py --net_basename vindr_13-lr:0.01-wd:0.0005_solar-sky-29 --model_name last


python3 vindr_ols.py --net_basename vindr_0-lr:0.01-wd:0.0005_spring-donkey-30 --model_name last --tasks 0 

python3 vindr_ols.py --net_basename vindr_1-lr:0.01-wd:0.0005_dandy-durian-17 --model_name last --tasks 1

python3 vindr_ols.py --net_basename vindr_2-lr:0.01-wd:0.0005_solar-violet-18 --model_name last --tasks 2

python3 vindr_ols.py --net_basename vindr_3-lr:0.01-wd:0.0005_silver-eon-19 --model_name last --tasks 3

python3 vindr_ols.py --net_basename vindr_4-lr:0.01-wd:0.0005_quiet-sun-20 --model_name last --tasks 4

python3 vindr_ols.py --net_basename vindr_5-lr:0.01-wd:0.0005_radiant-field-21 --model_name last --tasks 5

python3 vindr_ols.py --net_basename vindr_6-lr:0.01-wd:0.0005_silver-river-22 --model_name last --tasks 6

python3 vindr_ols.py --net_basename vindr_7-lr:0.01-wd:0.0005_bumbling-elevator-23 --model_name last --tasks 7

python3 vindr_ols.py --net_basename vindr_8-lr:0.01-wd:0.0005_fresh-capybara-24 --model_name last --tasks 8

python3 vindr_ols.py --net_basename vindr_9-lr:0.01-wd:0.0005_feasible-mountain-25 --model_name last --tasks 9

python3 vindr_ols.py --net_basename vindr_10-lr:0.01-wd:0.0005_unique-field-26 --model_name last --tasks 10

python3 vindr_ols.py --net_basename vindr_11-lr:0.01-wd:0.0005_stellar-morning-27 --model_name last --tasks 11

python3 vindr_ols.py --net_basename vindr_12-lr:0.01-wd:0.0005_dulcet-resonance-28 --model_name last --tasks 12

python3 vindr_ols.py --net_basename vindr_13-lr:0.01-wd:0.0005_solar-sky-29 --model_name last --tasks 13


