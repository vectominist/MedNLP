import csv
import warnings
import zipfile
import os

'''
****** AICUP spring 2021 check format ******
'''

'''
File extract
We check two things:
1. Upload file must be a .zip file
2. Contents need to contain two things : A. decision.csv B. qa.csv
'''
def file_extract():
    # Get the uploaded compressed file
    # file_list = os.listdir('./')
    target_zip = "results.zip"
    '''
    for i in file_list:
        if "zip" in i:
            target_zip = i

    # Check the archive format
    if target_zip == "":
        warnings.warn("Upload file must be a .zip file.")
        exit()
    '''
    # Extract zipfile
    zf = zipfile.ZipFile(target_zip, 'r')
    zf.extractall()

    # Check if there is a specified file
    if not os.path.isfile("./decision.csv"):
        warnings.warn("Risk file is not exist or wrong name.")
        exit(1)
    if not os.path.isfile("./qa.csv"):
        warnings.warn("QA file is not exist or wrong name.")
        exit(1)

'''
Risk classification
We need to put two path:
1. Check_format of data, must be a .csv type  (./check_format_data/check_format_decision.csv)
2. Predict of data, must be a .csv type (./decision.csv)

We check four things:
1. Predict id is from small to large
2. Predict label is smaller than 1
3. Number of predict and number of answer must be equal
4. Predict id and answer id must be same

We get one thing:
1. check if the format is correct
'''
def check_decision():
    # List for check, predict, flag
    risk_chk_id = []
    risk_chk_label = []
    risk_pred_id = []
    risk_pred_label = []
    flag = 1

    # Get risk check
    with open("./check_format_data/check_format_decision.csv", "r", encoding = "UTF-8") as risk_chk:
        for i, line in enumerate(csv.reader(risk_chk)):
            # Remove article_id, label
            if i == 0:
                continue
            id = int(line[0])
            label = line[1]

            # Record check
            risk_chk_id.append(id)
            risk_chk_label.append(float(label))

    # Get risk predict
    with open("decision.csv", "r", encoding= "UTF-8") as risk_pred:
        check_id_order = 0
        for i, line in enumerate(csv.reader(risk_pred)):
            # Remove article_id, label
            if i == 0:
                continue
            # Get id, label
            id = int(line[0])
            label = line[1]

            # Check id is arrange from small to large
            if check_id_order > id:
                warnings.warn("Risk predict id must be arrange from small to large.")
                exit(1)
            else:
                check_id_order = id

            # Make label from str to float, and check it must be small than 1
            label = float(label)
            if label > 1.0:
                warnings.warn("Label must be small than 1.")
                exit(1)

            risk_pred_id.append(id)
            risk_pred_label.append(float(label))

    # Check chk and pred length is same
    if len(risk_chk_label) != len(risk_pred_label):
        warnings.warn("Risk check length is not equal to predict length.")
        exit(1)

    # Check chk and pred id is all same
    for i in range(0, len(risk_chk_label)):
        if risk_chk_id[i] != risk_pred_id[i]:
            warnings.warn("Risk check id and predict id is not equal.")
            exit(1)

    return flag

'''
QA
We need to put two path:
1. Check_format of data, must be a .csv type  (./check_format_data/check_format_QA.csv)
2. Predict of data, must be a .csv type (./qa.csv)

We check five things:
1. Predict id is from small to large
2. Predict id's length is 1
3. Predict label must be 'A' or 'B' or 'C'
4. Number of predict and number of check must be equal
5. Predict id and check id must be same

We get one thing:
1. check if the format is correct
'''
def check_qa():
    # List for check, predict, real_id
    qa_chk_id = []
    qa_chk_label = []
    qa_pred_id = []
    qa_pred_label = []
    flag = 1

    # Get QA check
    with open("./check_format_data/check_format_QA.csv", "r", encoding = "UTF-8") as qa_chk:
        for i, line in enumerate(csv.reader(qa_chk)):
            # Remove article_id, label
            if i == 0:
                continue
            id = int(line[0])
            label = line[1]
            # Record check
            qa_chk_id.append(id)
            qa_chk_label.append(label)

    # Get QA predict
    with open("qa.csv", "r", encoding = "UTF-8") as qa_pred:
        check_id_order = 0
        for i, line in enumerate(csv.reader(qa_pred)):
            # Remove article_id, label
            if i == 0:
                continue
            # Get id, label
            id = int(line[0])
            label = line[1]

            # Check id is arrange from small to large
            if check_id_order > id:
                warnings.warn("QA predict id must be arrange from small to large.")
                exit(1)
            else:
                check_id_order = id
            # Check label is length 1
            if len(label) != 1:
                warnings.warn("QA predict label must be length 1, and remove any space.")
                print("Error id : ", id)
                exit(1)
            # Make full type to half type, and check for unreasonable answer
            if ("A" in label) or ("Ａ" in label):
                label = "A"
            elif ("B" in label) or ("Ｂ" in label):
                label = "B"
            elif ("C" in label) or ("Ｃ" in label):
                label = "C"
            else:
                warnings.warn("QA predict label must be 'A' or 'B' or 'C'.")
                print("Error id : ", id)
                exit(1)

            qa_pred_id.append(id)
            qa_pred_label.append(label)

    # Check chk and pred length is same
    if len(qa_chk_label) != len(qa_pred_label):
        warnings.warn("QA check length is not equal to predict length.")
        exit(1)
    # Check chk and pred id is all same
    for i in range(0, len(qa_chk_label)):
        if qa_chk_id[i] != qa_pred_id[i]:
            warnings.warn("QA check id and predict id is not equal.")
            exit(1)

    return flag

def main():
    # Extract file
    file_extract()
    # Get risk flag
    risk_flag = check_decision()
    # Get QA flag
    qa_flag = check_qa()
    # Pass
    if risk_flag == 1 and qa_flag == 1:
        print('Pass')
    return

if __name__ == "__main__":
    main()
