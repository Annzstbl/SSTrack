# 秦昊林训练代码及结果

CUDA_VISIBLE_DEVICES=3 nohup python -u tracking/train.py --script sstrack --config baseline_must --save_dir ./save/ --mode single > logs/0428.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python tracking/test.py sstrack baseline_must --dataset MUSTHSI
python tracking/analysis_results.py
MUSTHSI      | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
SSTrack      | 58.44      | 73.71      | 48.25      | 76.79        | 73.68             |


====================================================================================
CUDA_VISIBLE_DEVICES=3 nohup python -u tracking/train.py --script sstrack --config baseline_must_trans_enc_cope --save_dir ./save/ --mode single > logs/0509.log 2>&1 &




# 冯涛训练代码及结果

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


# 梯度全部回传

CUDA_VISIBLE_DEVICES=3 nohup python -u tracking/train.py \
  --script sstrack \
  --config baseline_must_trans_enc_cope_cvtp \
  --save_dir /data4/litianhao/must2 \
  --mode single \
  > logs/baseline_must_trans_enc_cope_cvtp.log 2>&1 &

## 结果
框几乎迅速就扩大到全图了，明显是有错误

CUDA_VISIBLE_DEVICES=3 python tracking/test.py sstrack baseline_must_trans_enc_cope_cvtp --dataset MUSTHSI --save_dir /data4/litianhao/must2 --threads 4
python tracking/analysis_results.py --tracker_param baseline_must_trans_enc_cope_cvtp
MUSTHSI                                        | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
sstrack_baseline_must_trans_enc_cope_cvtp      | 8.56       | 3.64       | 1.57       | 10.12        | 9.65              |




# ~~197 都没跑完~~

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


# ~~没训练完停止训练~~

CUDA_VISIBLE_DEVICES=3 nohup python -u tracking/train.py \
  --script sstrack \
  --config baseline_must_trans_enc_cope_cvtp_hard_223 \
  --save_dir /data4/litianhao/must2 \
  --mode single \
  > logs/baseline_must_trans_enc_cope_cvtp_hard_223.log 2>&1 &

# 测试梯度只回传到self.prompt，不回传到其他模块，用一个seq200iterations测试， 感觉正确，至少框不会迅速扩大到全图

 CUDA_VISIBLE_DEVICES=3 python -u tracking/train.py \
  --script sstrack \
  --config  baseline_must_trans_enc_cope_cvtp_one_seq \
  --save_dir /data4/litianhao/must2 \
  --mode single 

# 梯度只回传到self.prompt，不回传到其他模块
CUDA_VISIBLE_DEVICES=3 nohup python -u tracking/train.py \
--script sstrack \
--config baseline_must_trans_enc_cope_cvtp_2 \
--save_dir /data4/litianhao/must2 \
--mode single \
> logs/baseline_must_trans_enc_cope_cvtp_2.log 2>&1 &


# cvpt3
CUDA_VISIBLE_DEVICES=3 nohup python -u tracking/train.py \
--script sstrack \
--config baseline_must_trans_enc_cope_cvtp_3 \
--save_dir /data4/litianhao/must2 \
--mode single \
> logs/baseline_must_trans_enc_cope_cvtp_3.log 2>&1 &

推理和评测
CUDA_VISIBLE_DEVICES=3 python tracking/test.py sstrack baseline_must_trans_enc_cope_cvtp_3 --dataset MUSTHSI --save_dir /data4/litianhao/must2 --threads 4 && python tracking/analysis_results.py --tracker_param baseline_must_trans_enc_cope_cvtp_3

MUSTHSI                                          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
sstrack_baseline_must_trans_enc_cope_cvtp_3      | 48.98      | 60.33      | 38.28      | 63.73        | 60.51             |

# cvpt4
CUDA_VISIBLE_DEVICES=3 nohup python -u tracking/train.py \
--script sstrack \
--config baseline_must_trans_enc_cope_cvtp_4 \
--save_dir /data4/litianhao/must2 \
--mode single \
> logs/baseline_must_trans_enc_cope_cvtp_4.log 2>&1 &

推理和评测
CUDA_VISIBLE_DEVICES=3 python tracking/test.py sstrack baseline_must_trans_enc_cope_cvtp_4 --dataset MUSTHSI --save_dir /data4/litianhao/must2 --threads 4 && python tracking/analysis_results.py --tracker_param baseline_must_trans_enc_cope_cvtp_4
MUSTHSI                                          | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
sstrack_baseline_must_trans_enc_cope_cvtp_4      | 47.88      | 58.22      | 38.74      | 61.70        | 59.26             |


# 再评测一遍冯涛
CUDA_VISIBLE_DEVICES=3 python tracking/test.py sstrack baseline_must_trans_enc_cope --dataset MUSTHSI --save_dir /data4/litianhao/must2 --threads 1
python tracking/analysis_results.py --tracker_param baseline_must_trans_enc_cope

MUSTHSI                                   | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
sstrack_baseline_must_trans_enc_cope      | 62.79      | 78.47      | 53.75      | 82.36        | 79.66             |