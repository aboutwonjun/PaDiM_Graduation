#Layer 별 실험! 

# python main_per_layer_time.py --arch resnet50; #이건 일단 돌아가긴 함.. 
# python main_per_layer_time.py  --arch fcn_resnet50;

# python main_per_layer_time.py --arch resnet50 --layer layer1;
# python main_per_layer_time.py --arch fcn_resnet50 --layer layer1;

# python main_per_layer_time.py --arch resnet50 --layer layer2;
# python main_per_layer_time.py --arch fcn_resnet50 --layer layer2;

# python main_per_layer_time.py --arch resnet50 --layer layer3;
# python main_per_layer_time.py --arch fcn_resnet50 --layer layer3;

# python main_per_layer_time.py --arch resnet50 --layer layer1 layer2;
# python main_per_layer_time.py --arch fcn_resnet50 --layer layer1 layer2;

# python main_per_layer_time.py --arch resnet50 --layer layer2 layer3;
# python main_per_layer_time.py --arch fcn_resnet50 --layer layer2 layer3;

# python main_per_layer_time.py --arch resnet50 --layer layer1 layer3;
# python main_per_layer_time.py --arch fcn_resnet50 --layer layer1 layer3;



#layer 1 
=========================
# full
# python main_wideresnet50_ori.py --arch wide_resnet50_2 --layer layer1 --selection full --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer1_full;
# # python main_wideresnet50.py --arch wide_resnet50_2 --layer layer2 --selection full --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer2_full;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer3 --selection full --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer3_full;

# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 layer2 --selection full --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer12_full;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer2 layer3 --selection full --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer23_full;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 layer3 --selection full --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer13_full;


# # random
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 --selection random --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer1_random;
# # python main_wideresnet50.py --arch wide_resnet50_2 --layer layer2 --selection random --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer2_random;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer3 --selection random --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer3_random;

# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 layer2 --selection random --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer12_random;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer2 layer3 --selection random --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer23_random;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 layer3 --selection random --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer13_random;

# # var large 
python main_per_layer_plotter_large copy.py --arch resnet50 --layer layer1 --selection var_large;


# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer2 --selection var_large --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer2_var_large;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer3 --selection var_large --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer3_var_large;

# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 layer2 --selection var_large --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer12_var_large;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer2 layer3 --selection var_large --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer23_var_large;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 layer3 --selection var_large --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer13_var_large;

# # var small
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 --selection var_small --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer1_var_small;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer2 --selection var_small --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer2_var_small;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer3 --selection var_small --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer3_var_small;

# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 layer2 --selection var_small --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer12_var_small;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer2 layer3 --selection var_small --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer23_var_small;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 layer3 --selection var_small --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer13_var_small;

# # mean large 
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 --selection mean_large --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer1_mean_large;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer2 --selection mean_large --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer2_mean_large;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer3 --selection mean_large --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer3_mean_large;

# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 layer2 --selection mean_large --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer12_mean_large;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer2 layer3 --selection mean_large --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer23_mean_large;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 layer3 --selection mean_large --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer13_mean_large;

# # mean large 
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 --selection mean_small --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer1_mean_small;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer2 --selection mean_small --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer2_mean_small;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer3 --selection mean_small --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer3_mean_small;

# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 layer2 --selection mean_small --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer12_mean_small;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer2 layer3 --selection mean_small --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer23_mean_small;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 layer3 --selection mean_small --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer13_mean_small;

# # hybrid 
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 --selection hybrid --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer1_hybrid;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer2 --selection hybrid --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer2_hybrid;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer3 --selection hybrid --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer3_hybrid;

# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 layer2 --selection hybrid --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer12_hybrid;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer2 layer3 --selection hybrid --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer23_hybrid;
# python main_wideresnet50.py --arch wide_resnet50_2 --layer layer1 layer3 --selection hybrid --save_path ./mvtec_result_weight_test_wide_resnet50_2_layer13_hybrid;



















# #FULL
# CUDA_VISIBLE_DEVICES=1 python main_per_layer.py \
#     --arch resnet18 \
#     --save_path ./mvtec_result_weight_test_resnet18_full \
#     --selection full;
# #RANDOM
# CUDA_VISIBLE_DEVICES=1 python main_per_layer.py \
#     --arch resnet18 \
#     --save_path ./mvtec_result_weight_test_resnet18_random \
#     --selection random;