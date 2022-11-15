# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Baseline UDA
uda = dict(
    type='GAN',
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    print_grad_magnitude=False,
    # Discriminator
    power=0.9,
    lr_dis=1e-4,
    px_wo_cls_adv_lambda=0.01
)
use_ddp_wrapper = True
