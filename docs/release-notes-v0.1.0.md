首个本地优先版本，面向课程、访谈、播客、个人记录和字幕等通用音视频转录场景。

- FunASR 模块化模型注册：Paraformer-zh、SenseVoiceSmall、FSMN-VAD、CT-Punc、CAM++。
- CPU离线推理、短窗说话人聚类、分声道与长媒体分块、可追溯后处理。
- TXT/SRT/VTT/JSON/CSV/Markdown/DOCX、学习句子与练习字段、任务去重、完整性校验与断点恢复。
- 独立Python环境、一键安装、可复用Codex Skill；本机Codex发现检查通过。

验证：75项软件测试通过，19项真实合成媒体集成场景执行完成；第二独立虚拟环境安装和CPU推理通过。详细结果见 docs/tests-report.md。

限制：自动人数在双声线样本中估为4簇，仍属实验功能，建议显式指定已知人数。合成声线标签匹配不等于真实访谈准确率；小时级真实录音和其他物理平台尚未验证。重叠语音源分离和自动真实身份识别不支持。

代码与Skill为MIT；模型权重许可独立，SenseVoice使用自定义模型许可。完整ASR请使用源码仓库的install.sh/uv frozen流程，排除kaldiio评估许可依赖；Kaldi archive功能明确禁用。wheel仅分发基础CLI，不是完整ASR运行时。未打包任何媒体、逐字稿、声纹向量、模型权重或密钥。
