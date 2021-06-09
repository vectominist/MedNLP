MedNLP
==============

### Environment

* Python 3.8.5
* Pytorch 1.8.1
* Transfomrers 4.6.1

```bash
pip install -r requirement.txt
```

### Download Data

```bash
vim aidea-web.tw_cookies.txt # put log-in cookies.txt here
./download.sh
```

### Risk Evaluation

##### Model

See `report.pdf`

##### Train Masked Language Model (MLM)

```bash
vim config/mlm.yaml # Set train path and pretrained model.
./trian_mlm.sh
```

##### Train Risk Model

```bash
vim config/risk.yaml # Put your pretrained MLM here. Set train path and test path.
./train_risk.sh
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
./run_qa_rule_base.sh
```
