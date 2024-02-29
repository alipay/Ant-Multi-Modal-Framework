# 模型使用
## 使用说明

```
# 新建环境（Python版本3.8）
conda create -n m2-encoder python=3.8
source activate m2-encoder

# clone项目地址
cd /YourPath/
git clone https://github.com/alipay/Ant-Multi-Modal-Framework

# 安装包依赖
cd ./Ant-Multi-Modal-Framework/prj/M2_Encoder/
pip install -r requirements.txt

# 运行demo
python run.py
```

## 参考

Github: [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3).

Paper: [VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts](https://arxiv.org/abs/2111.02358).