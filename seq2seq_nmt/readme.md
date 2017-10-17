## 利用 seq2seq 实现机器翻译

* 翻译模型:
  - 语料
    + 来源: [United Nations Parallel Corpus](https://conferences.unite.un.org/UNCorpus)
    + 筛选和处理: [data.ipynb](data.ipynb), 辅助函数 [data_helpers.py](data_helpers.py)
  - 模型训练和测试: [seq2seq_nmt.ipynb](seq2seq_nmt.ipynb)
  - 封装的类: [nmt_model.py](nmt_model.py) (暂时只用来 restore 和 decode)
* 接入 API 和微信机器人
  * 用 `flask-restful` 创建翻译 API: [api.py](api.py)
  * 用 `wxpy` 实现微信机器人: [bot.py](bot.py)
