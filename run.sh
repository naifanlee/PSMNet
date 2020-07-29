###
 # @Author: your name
 # @Date: 2020-06-15 17:31:47
 # @LastEditTime: 2020-06-15 17:33:05
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /ai/depth_map/PSMNet/PSMNet/run.sh
### 
# train
python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath /home/nfli/dataset/dataset_softlink/SceneFlow/ \
               --epochs 300 \
               --savemodel /home/nfli/ai/depth_map/PSMNet/PSMNet/save_model/SceneFlow

# finetune
python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype 2015 \
                   --datapath ~/dataset/KITTI/Stereo2015/training/\
                   --epochs 600 \
                   --loadmodel /home/nfli/ai/depth_map/PSMNet/PSMNet/save_model/SceneFlow/checkpoint_9_2900.tar \
                   --savemodel /home/nfli/ai/depth_map/PSMNet/PSMNet/save_model/KITTI2015/

# eval
python submission.py --maxdisp 192 \
                     --model stackhourglass \
                     --KITTI 2015 \
                     --datapath /home/nfli/dataset/KITTI/Stereo2015/testing/ \
                     --loadmodel /home/nfli/ai/depth_map/PSMNet/PSMNet/save_model/KITTI2015/KITTI2015finetune_300_0.tar \
                     --outpath /home/nfli/ai/depth_map/PSMNet/PSMNet/dataset/
# test
python Test_img.py --loadmodel /home/nfli/ai/depth_map/PSMNet/PSMNet/save_model/KITTI2015/KITTI2015finetune_300_0.tar --leftimg  --rightimg 

# eval test
python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype 2015 \
                   --datapath ~/dataset/KITTI/Stereo2015/training/\
                   --epochs 600 \
                   --loadmodel /home/nfli/ai/depth_map/PSMNet/PSMNet/save_model/KITTI2015/KITTI2015finetune_300_0.tar \
                   --savemodel /home/nfli/ai/depth_map/PSMNet/PSMNet/save_model/KITTI2015_2/
