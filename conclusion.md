1. self.prompt不训练    track_query更新     采用add_token形式   AUC=63.2
2. self.prompt不训练    track_query=None    采用add_token形式   AUC=62.8  采用1的权重
3. self.prompt训练      track_query=更新    采用cat_token形式   AUC=49
4. self.prompt训练      track_query=更新    采用cat_token形式   AUC=48

这是目前实验的一个统计结果，需要你仔细思考模型的工作情况，我的理解是self.prompt在不训练的时候，它实际上生成的只是当前帧信息的一个固定压缩模式，主要起作用的还是add_token，因为即使track_query=None的时候仍然能保持很好的跟踪结果。
回到我们的实验中，如果self.prompt训练，代表着self.prompt需要在训练的过程中确定什么样的输出才能接续到后续时间序列中，表示self.prompt会发挥作用。
因此你不能够得出结论说aplha=0就是一种保守的方案，因为可以稳定的传播track_query和前序实验主要通过cls_token的观点是相反的。
需要你仔细剖析实验结果和网络结构，找到问题的根本。