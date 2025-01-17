# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=0.0,
    by_epoch=False)
