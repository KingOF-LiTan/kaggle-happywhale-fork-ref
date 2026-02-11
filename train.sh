python -m run.train --config-name config_effb0 \
  preprocessing.h_resize_to=384 preprocessing.w_resize_to=384 \
  training.batch_size=128 training.num_workers=8 \
  training.epoch=15 \
  training.use_amp=true \
  +augmentation.p_sharpen=0.3 augmentation.p_gray=0.2 \
  +forwarder.mixup_alpha=0.5 \
  dataset.bbox=fb dataset.crop=null \
  +freeze_bn_stats=true \
  out_dir=./outputs/body_b0_for_refinement