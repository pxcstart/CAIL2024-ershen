# CAIL2024-ershen
#### 基于Qwen大模型的智能二审改判检索问答

使用的模型：Qwen1.5-7B-Chat (注:尝试了一些垂域大模型，例如Disc-Law、LLM-Chat，发现效果都不是很好，所以还是用通用域大模型解决该任务); 中文文本嵌入模型M3E-base

显卡：建议使用32GB显存以上的显卡（v100,a100等)，或多卡RTX 4090

原始数据集和微调后的模型文件放在了[google云盘](https://drive.google.com/drive/folders/1D5QoZ6XRr60McqQ9W9oe-Y9847wSzXoQ?usp=sharing)，建议直接使用axolotl微调的模型文件（axolotl下的checkpoint-488)进行推理

下面简单介绍下Solution：

<Method>首先对源数据集进行清洗，设计提示词对案件的一审内容和二审内容进行归纳总结，并分成案件基本信息和法院判决结果两部分（详见summary_clean.py）；接下来对模型进行微调，微调任务为根据一审的基本信息和判决结果推测二审的判决结果(详见finetune.py)。在推理阶段，采用检索-生成范式作答，首先从cases数据库中根据query的案件基本信息检索top-k相似案件，将其一审结果和二审结果作为知识馈送给大模型，设计相应的提示词让大模型根据指令进行推理（详见inference.py）。

微调框架：可以使用finetune.py进行微调，或直接使用axolotl框架零代码微调

