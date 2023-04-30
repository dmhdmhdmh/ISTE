
# # ***将data_path修改为自己的data路径***
# 我的data结构
# --|fastmri
#   |---|knee_singlecoil
#   |   |---|singlecoil_test
#   |       |singlecoil_train
#   |       |singlecoil_val
#   |---|other

###单卡需要修改为0
export PYTHONPATH=/data/minghongduan/githubsave/ISTE

python train.py

#cd fastmri_examples/unet
#训练好的权重需要先运行run_pretrained_unet_inference.py下载
#python run_pretrained_unet_inference.py --data_path /home/miccai/code/python/Data/fastmri/knee_singlecoil_val/singlecoil_val \
#                                       --output_path ./output --challenge unet_knee_sc

#cd unet_reproduce_20201111.py
#for train
#python unet_knee_sc_leaderboard.py --mask_type random --challenge singlecoil \
#                                   --data_path /home/miccai/code/python2/Data/fastmri/knee_singlecoil

#for val, 只显示指标.需要用自己的路径替换eval_ckpt路径
#python unet_knee_sc_leaderboard.py --mode val --test_split val --challenge singlecoil \
#                                   --data_path /home/miccai/code/python2/Data/fastmri/knee_singlecoil \
#                                   --eval_ckpt /home/miccai/code/python/fastMRI/fastmri_examples/unet/knee_sc_leaderboard_state_dict.pt

#for testval,可以生成重建图像.需要用自己的路径替换eval_ckpt路径
#python unet_knee_sc_leaderboard.py --mode test --test_split val --challenge singlecoil \
#                                   --data_path /home/miccai/code/python2/Data/fastmri/knee_singlecoil \
#                                   --eval_ckpt /home/miccai/code/python/fastMRI/fastmri_examples/unet/knee_sc_leaderboard_state_dict.pt