conda activate must


CUDA_VISIBLE_DEVICES=3 nohup python -u tracking/train.py --script sstrack --config baseline_must --save_dir ./save/ --mode single > logs/0428.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python tracking/test.py sstrack baseline_must --dataset MUSTHSI
python tracking/analysis_results.py
MUSTHSI      | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
SSTrack      | 58.44      | 73.71      | 48.25      | 76.79        | 73.68             |


====================================================================================
CUDA_VISIBLE_DEVICES=3 nohup python -u tracking/train.py --script sstrack --config baseline_must_trans_enc_cope --save_dir ./save/ --mode single > logs/0509.log 2>&1 &

ln -s /data3/fengtao/pretrained_networks/mae_pretrain_vit_base_cope.pth /data/users/qinhaolin01/SSTrack-fengtao/pretrained_networks/mae_pretrain_vit_base_cope.pth

CUDA_VISIBLE_DEVICES=3 nohup python -u tracking/train.py \
  --script sstrack \
  --config baseline_must_trans_enc_cope \
  --save_dir /data4/litianhao/must2 \
  --mode single \
  > logs/resume.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 python tracking/test.py sstrack baseline_must_trans_enc_cope --dataset MUSTHSI --save_dir /data4/litianhao/must2 --threads 4
python tracking/analysis_results.py --tracker_param baseline_must_trans_enc_cope

MUSTHSI                                   | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
sstrack_baseline_must_trans_enc_cope      | 63.19      | 79.08      | 54.48      | 82.44        | 79.29             |


CUDA_VISIBLE_DEVICES=3 nohup python -u tracking/train.py \
  --script sstrack \
  --config baseline_must_trans_enc_cope_cvtp \
  --save_dir /data4/litianhao/must2 \
  --mode single \
  > logs/baseline_must_trans_enc_cope_cvtp.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 nohup python -u tracking/train.py \
  --script sstrack \
  --config baseline_must_trans_enc_cope_cvtp \
  --save_dir /data4/litianhao/must2 \
  --mode multiple \
  --nproc_per_node 2 \
  > logs/baseline_must_trans_enc_cope_cvtp.log 2>&1 &


  # 197
mkdir logs
CUDA_VISIBLE_DEVICES=0,1,2 nohup python -u tracking/train.py \
  --script sstrack \
  --config baseline_must_trans_enc_cope_cvtp \
  --save_dir /data4/litianhao/must2 \
  --mode multiple \
  --nproc_per_node 3 \
  > logs/baseline_must_trans_enc_cope_cvtp.log 2>&1 &

  CUDA_VISIBLE_DEVICES=2,3 python -u tracking/train.py \
  --script sstrack \
  --config baseline_must_trans_enc_cope_cvtp \
  --save_dir /data4/litianhao/must2 \
  --mode multiple \
  --nproc_per_node 2
