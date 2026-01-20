# training single GPU @ win or @ ubuntu
python train_stereo_la.py --config configs/config_full.toml
# training multi GPU @ ubuntu
torchrun --nproc_per_node=4 --master_port=29501 train_stereo_la.py --launcher pytorch --config configs/config_full.toml
torchrun --nproc_per_node=3 --master_port=29501 train_stereo_la.py --launcher pytorch --config configs/config_full.toml
torchrun --nproc_per_node=2 --master_port=29501 train_stereo_la.py --launcher pytorch --config configs/config_full.toml

# for multi traing with win
set USE_LIBUV=0
torchrun --nproc_per_node=3 --master_port=29501 --standalone --rdzv_backend=c10d --rdzv_conf "use_libuv=0" train_stereo_la.py --launcher pytorch --config configs/config_full.toml

# check training status
tensorboard --logdir outputs/run_debug

torchrun --nproc_per_node=2 --master_port=29501  train_stereo_la_with_instance_seg.py --launcher pytorch --config configs/config_full_panoptic.toml
torchrun --nproc_per_node=4 --master_port=29501  train_stereo_la_with_instance_seg.py --launcher pytorch --config configs/config_full_panoptic.toml