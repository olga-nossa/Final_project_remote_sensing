# Model comparison report

Device used: `cuda`

Cache reused: `True`

## Model parameters

```text
                 model_name architecture    data_source                                                                                                      checkpoint  base_channels  patch_h  patch_w  stride_y  stride_x  min_valid_ratio                       mean                        std  val_selected_threshold  best_test_threshold_diagnostic
     attention_unet_mars_v1 attention_bn           mars          C:\Users\nicof\Desktop\proy_olga\Comparacion_de_metodos\Modelos generados\best_attention_unet_mars.pth             16      256      256       256       256              0.7 0.550000,0.350000,0.250000 0.200000,0.150000,0.150000                    0.55                            0.55
     attention_unet_mars_v2 attention_gn           mars       C:\Users\nicof\Desktop\proy_olga\Comparacion_de_metodos\Modelos generados\best_attention_unet_mars_v2.pth             32      256      256       128       128              0.7 0.253318,0.174194,0.175665 0.228469,0.173787,0.179681                    0.60                            0.60
attention_unet_multiyear_v1 attention_bn  multiyear_raw     C:\Users\nicof\Desktop\proy_olga\Comparacion_de_metodos\Modelos generados\best_attention_unet_multiyear.pth             16      256      256       256       256              0.7 0.550000,0.350000,0.250000 0.200000,0.150000,0.150000                    0.50                            0.50
attention_unet_multiyear_v2 attention_bn  multiyear_raw  C:\Users\nicof\Desktop\proy_olga\Comparacion_de_metodos\Modelos generados\best_attention_unet_multiyear_v2.pth             16      256      256       256       256              0.7 0.550000,0.350000,0.250000 0.200000,0.150000,0.150000                    0.70                            0.65
    cbam_unet_article_style         cbam multiyear_cbam C:\Users\nicof\Desktop\proy_olga\Comparacion_de_metodos\Modelos generados\best_cbam_unet_mdad_article_style.pth             16      668      688       334       334              0.5 0.454880,0.339112,0.324565 0.161970,0.126247,0.154244                    0.75                            0.60
```

## Test metrics

```text
                 model_name  threshold  accuracy  precision   recall       f1      iou          tn         fp         fn        tp
     attention_unet_mars_v1       0.55  0.890785   0.246474 0.364282 0.294016 0.172344  13588049.0  1088359.0   621259.0  355996.0
     attention_unet_mars_v2       0.60  0.921598   0.347188 0.345652 0.346418 0.209496  52608356.0  2281591.0  2297121.0 1213429.0
attention_unet_multiyear_v1       0.50  0.896275   0.070075 0.294057 0.113180 0.059984 202275727.0 19970469.0  3612811.0 1504896.0
attention_unet_multiyear_v2       0.70  0.896494   0.074344 0.314248 0.120241 0.063966 202222051.0 20024145.0  3509479.0 1608228.0
    cbam_unet_article_style       0.75  0.946771   0.108522 0.146919 0.124835 0.066573 542063104.0 17927071.0 12671576.0 2182320.0
```

