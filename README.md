MedNLP
==============
🏥 📖 Mandarin medical dialogue analysis implemented in PyTorch.

### Introduction

This is a repository for analyzing mandarin medical dialgoues, including risk assessment and multiple-choice questing answering. The codes are designed for the [Medical Dialog Analysis Competition](https://aidea-web.tw/topic/3665319f-cd5d-4f92-8902-00ebbd8e871d). To reproduce our results, please refer to the [reproduce](#reproduce) section.


### Installation

* Python 3.6+

```bash
pip install -r requirement.txt
```

### Download Data

```bash
vim aidea-web.tw_cookies.txt # put log-in cookies.txt here
./download.sh
```

### Reproduce

```bash
bash reproduce.sh config/risk.yaml config/qa.yaml config/
```

### Risk Evaluation

##### Model

See `report.pdf`

##### Train Masked Language Model (MLM)

```bash
vim config/mlm.yaml # Set train path and pretrained model.
bash trian_mlm.sh
```

##### Train Risk Model

```bash
vim config/risk.yaml # Put your pretrained MLM here. Set train path and test path.
bash train_risk.sh
```

##### Test

```bash
python src/train_risk.py --config config/risk.yaml --mode test
```

### Question Answering

##### Model

See `report.pdf`

##### Test

```bash
bash run_qa_rule_base.sh
```
