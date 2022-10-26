
# center_aprser_part5

[![Apach2.0 License](http://img.shields.io/badge/license-Apache-blue.svg?style=flat)](LICENSE)

センター試験英語筆記筆記試験大問5のソルバー（2017, 2016年度本試験対応）

---

## Features
ロボットは東大に入れるかプロジェクトで開発された，[センター試験XMLデータ](https://21robot.org/dataset.html)のうち，英語筆記問題の大問5を解くことができる．
対応年度は2017, 2016年度である．

---

## Configuration
動作環境
|  ライブラリ  |  バージョン  |
| ---- | ---- |
|  Anaconda  |    |
|  Python  |  3.6  |
|  CUDA  |  10.2  |
|  Pytorch  |  1.1.0  |

---

# Install
以下の手順で必要なデータセットとライブラリを準備してください．

```
# RACEデータセットをダウンロード
$ wget http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz -P /tmp
# センター試験XMLデータをダウンロード
$ wget https://21robot.org/data/center-devtest.tar.gz -P /tmp

$ tar -zxvf RACE.tar.gz -C data
$ mkdir data/center-2017-2015/dev && cp center-devtest/Eigo/Center-2017--Main-Eigo_hikki.xml 

$ rm /tmp/RACE.tar.gz && rm /tmp/center-devtest.tar.gz

# 必要なライブラリをインストール
$ conda env create -f environment.yml
```

---

## Usage
BERTの事前学習モデルをRACEデータセットを用いてfinetuneします．

```
./run.sh
```

2017年度センター試験英語筆記本試験大問５の評価

```
./eval.sh
```

---

## License
GitHub Changelog Generator is released under the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).

---

## Acknowledgements
This repository is based on [BERT-RACE](https://github.com/NoviScl/BERT-RACE)

---

## Feedback 
Any questions or suggestions?

You are welcome to discuss it on:

[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/dancing_nanachi)
---




