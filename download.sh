# !/bin/bash

cookie=aidea-web.tw_cookies.txt
if [ ! -f $cookie ]; then
	echo "Error: \"aidea-web.tw_cookies.txt\" not found! Login https://aidea-web.tw by chrome and write cookie into \"aidea-web.tw_cookies.txt\" first!"
	exit 1
fi
wget_opts="-nc --load-cookies $cookie"

mkdir -p data
wget ${wget_opts} -O data/SampleData_QA.json https://aidea-web.tw/file/3665319f-cd5d-4f92-8902-00ebbd8e871d-1614681891043555_train_test_dataset_1___SampleData_QA.json
wget ${wget_opts} -O data/SampleData_RiskClassification.csv https://aidea-web.tw/file/3665319f-cd5d-4f92-8902-00ebbd8e871d-1614681891043555_train_test_dataset_1___SampleData_RiskClassification.csv
wget ${wget_opts} -O data/Train_qa_ans.json https://aidea-web.tw/file/3665319f-cd5d-4f92-8902-00ebbd8e871d-1618969704984840_train_test_dataset_1___Train_qa_ans.json
wget ${wget_opts} -O data/Train_risk_classification_ans.csv https://aidea-web.tw/file/3665319f-cd5d-4f92-8902-00ebbd8e871d-1618969704984840_train_test_dataset_1___Train_risk_classification_ans.csv

[ ! -d data/Baseline ] && \
wget ${wget_opts} -O data/Baseline.zip https://aidea-web.tw/file/3665319f-cd5d-4f92-8902-00ebbd8e871d-1617075668876673_train_test_dataset_1___Baseline.zip && \
unzip -o data/Baseline.zip -d data/ && \
rm -rf data/Baseline.zip

if [ -d model ]; then
	echo "model folder exists, skipped..."
else
	mkdir model
	model=biobert_large_v1.1_pubmed.tar.gz
	gdown -O model/$model https://drive.google.com/u/0/uc?id=1GJpGjQj6aZPV-EfbiQELpBkvlGtoKiyA || exit 1
	tar --directory=model -xvf model/$model || exit 1
	rm -rf model/$model
fi
