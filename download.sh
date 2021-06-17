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
wget ${wget_opts} -O data/Develop_QA.json https://aidea-web.tw/file/3665319f-cd5d-4f92-8902-00ebbd8e871d-1621298399471774_train_test_dataset_1___Develop_QA.json
wget ${wget_opts} -O data/Develop_risk_classification.csv https://aidea-web.tw/file/3665319f-cd5d-4f92-8902-00ebbd8e871d-1621298399471774_train_test_dataset_1___Develop_risk_classification.csv
# wget ${wget_opts} -O data/Contestant_check_format.zip https://aidea-web.tw/file/3665319f-cd5d-4f92-8902-00ebbd8e871d-1623905483304198_train_test_dataset_1___Contestant_check_format.zip
wget ${wget_opts} -O data/Test_QA.json https://aidea-web.tw/file/3665319f-cd5d-4f92-8902-00ebbd8e871d-1623905483304198_train_test_dataset_1___Test_QA.json
wget ${wget_opts} -O data/Test_risk_classification.csv https://aidea-web.tw/file/3665319f-cd5d-4f92-8902-00ebbd8e871d-1623905483304198_train_test_dataset_1___Test_risk_classification.csv

# unzip -n data/Contestant_check_format.zip -d .

cp material/Develop_QA_human.json data

medlm=data/med_lm.txt
echo -n "" > $medlm
for file in Train_risk_classification_ans.csv Develop_risk_classification.csv; do
	cut -d ',' -f 3 data/$file | tail -n +2 >> $medlm
done

[ ! -d data/Baseline ] && \
wget ${wget_opts} -O data/Baseline.zip https://aidea-web.tw/file/3665319f-cd5d-4f92-8902-00ebbd8e871d-1617075668876673_train_test_dataset_1___Baseline.zip && \
unzip -o data/Baseline.zip -d data/ && \
rm -rf data/Baseline.zip

