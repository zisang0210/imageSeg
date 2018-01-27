# 设置目录，避免module找不到的问题
export PYTHONPATH=$PYTHONPATH:/home/zisang/objDetect:/home/zisang/objDetect/slim:/home/zisang/objDetect/object_detection

python convert_fcn_dataset.py --data_dir=/home/zisang/PASCAL\ VOC/VOCdevkit/VOC2012/ --output_dir=./

python train.py --checkpoint_path=/home/zisang/Documents/lesson9/vgg16-ckpt/vgg_16.ckpt \
 	--output_dir=../output \
 	--dataset_train=./fcn_train.record \
 	--dataset_val=./fcn_val.record \
 	--batch_size=16 \
 	--max_steps 2000