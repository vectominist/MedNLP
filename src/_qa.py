import torch
import json
import re
import numpy as np
import random
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from model import qa_model_hugginggface
from dataset import all_dataset
from torch.utils.tensorboard import SummaryWriter
from data import QADataset

'''
Here is the part of training the model, you need to set hyperparameters by yourself.
'''

def qa_train():
    # Hyperparameters
    args = {
        "batch_size": 2,
        "learning_rate": 1e-3,
        "random_seed": 42,
        "n_epoch": 50,
        "n_fine_tune_huggingface_epoch": 40,
        "log_step": 15,
        "save_step": 3000,
        "d_emb": 300,
        "n_cls_layers": 2,
        "p_drop": 0.0,
        "weight_decay": 0.0,
        "model_path": os.path.join("exp", "_qa", "_1"),
        "log_path": os.path.join("log", "_qa", "_1"),
        "qa_data": os.path.join("data", "Train_qa_ans.json"),
        "risk_data": os.path.join("data", "Train_risk_classification_ans.csv"),
        "max_q_len": 20,
        "max_c_len": 20,
        "max_doc_len": 170, # number of sentences per document
        "max_sent_len": 70,
    }

    # Save training configuration
    if not os.path.isdir(args["model_path"]):
        os.makedirs(args["model_path"])
    with open(os.path.join(args["model_path"], "cfg.json"), "w") as f:
        json.dump(args, f)

    # Random seed
    random_seed = args["random_seed"]
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Model
    model = qa_model_hugginggface(args["n_cls_layers"], args["p_drop"])
    model = model.train()
    model = model.to(device)

    # Remove weight decay on bias and layer-norm.
    no_decay = ['bias', 'LayerNorm.weight']
    optim_group_params = [
        {
            'params': [
                param for name, param in model.named_parameters()
                if not any(nd in name for nd in no_decay)
            ],
            'weight_decay': args["weight_decay"],
        },
        {
            'params': [
                param for name, param in model.named_parameters()
                if any(nd in name for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args["learning_rate"])

    # Data
    # data = all_dataset(args["qa_data"], args["risk_data"])
    data = QADataset(args["qa_data"], 'train', max_q_len=args["max_q_len"], max_c_len=args["max_c_len"],
                 max_doc_len=args["max_doc_len"], max_sent_len=args["max_sent_len"])
    dataldr = torch.utils.data.DataLoader(data, batch_size=args["batch_size"], shuffle=True)

    # Log writer
    if not os.path.isdir(args["log_path"]):
        os.makedirs(args["log_path"])
    writer = SummaryWriter(log_dir=args["log_path"])

    # Train loop
    step = 0
    avg_loss = 0
    for epoch in range(args["n_epoch"]):
        tqdm_dldr = tqdm(dataldr)
        if epoch == args["n_fine_tune_huggingface_epoch"] - 1:
            model.EnableEncoderTrain()
        for batch_data in tqdm_dldr:
            optimizer.zero_grad()
            # batch_document = []

            # for idx in batch_data["article_id"]:
            #     batch_document.append(data.article[idx])

            # batch_document = torch.LongTensor(batch_document).to(device)
            # batch_question = batch_data["question"].to(device)
            # batch_choice = batch_data["choice"].to(device)
            # batch_qa_answer = batch_data["qa_answer"].float().to(device)
            batch_document_id = batch_data[0]
            batch_document = batch_data[1].to(device)
            batch_question = batch_data[2].to(device)
            batch_choice = batch_data[3].to(device)
            batch_qa_answer = batch_data[4].to(device)

            loss = model.loss_fn(batch_document, batch_question,
                                 batch_choice, batch_qa_answer)
            loss.backward()
            optimizer.step()

            step = step + 1
            avg_loss = avg_loss + loss

            if step % args["log_step"] == 0:
                avg_loss = avg_loss/args["log_step"]
                tqdm_dldr.set_description(f"epoch:{epoch}, loss:{avg_loss}")
                writer.add_scalar("loss", avg_loss, step)
                avg_loss = 0

            if step % args["save_step"] == 0:
                if not os.path.isdir(args["model_path"]):
                    os.makedirs(args["model_path"])
                torch.save(model.state_dict(), os.path.join(
                    args["model_path"], f"model-{step}.pt"))

    if not os.path.isdir(args["model_path"]):
        os.makedirs(args["model_path"])

    torch.save(model.state_dict(), os.path.join(args["model_path"], f"model-{step}.pt"))


def qa_test(exp_path: str, eval_set="val"):
    # Hyperparameters
    batch_size = 8
    with open(os.path.join(exp_path, "cfg.json"), "r") as f:
        args = json.load(f)

    # Device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Load checkpoints
    ckpts = []
    for ckpt in os.listdir(args["model_path"]):
        match = re.match(r'model-(\d+).pt', ckpt)
        if match is None:
            continue
        ckpts.append(int(match.group(1)))
    ckpts = sorted(ckpts)

    # Log writer
    writer = SummaryWriter(log_dir=args["log_path"])

    # Evaluate on training set
    print("evaluate on %s set..." % eval_set)

    # Data
    # data = all_dataset(args["qa_data"], args["risk_data"])
    data = QADataset(args["qa_data"], eval_set, max_q_len=args["max_q_len"], max_c_len=args["max_c_len"],
                 max_doc_len=args["max_doc_len"], max_sent_len=args["max_sent_len"])

    for ckpt in ckpts:
        # Model
        model = qa_model_hugginggface(args["n_cls_layers"], 0.0)
        model.load_state_dict(torch.load(os.path.join(
            args["model_path"], f"model-{ckpt}.pt")))
        model = model.eval()
        model = model.to(device)

        dataldr = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
        tqdm_dldr = tqdm(dataldr)
        answer = {"qa": []}
        pred = {"qa": []}

        for batch_data in tqdm_dldr:
            # batch_document = []

            # for idx in batch_data["article_id"]:
            #     batch_document.append(data.article[idx])

            # batch_document = torch.LongTensor(batch_document).to(device)
            # batch_question = batch_data["question"].to(device)
            # batch_choice = batch_data["choice"].to(device)
            batch_document_id = batch_data[0]
            batch_document = batch_data[1].to(device)
            batch_question = batch_data[2].to(device)
            batch_choice = batch_data[3].to(device)
            batch_answer = batch_data[4]

            answer["qa"] = answer["qa"] + batch_answer.argmax(dim=-1).tolist()

            pred_qa = model(batch_document, batch_question, batch_choice)

            pred["qa"] = pred["qa"] + pred_qa.argmax(dim=-1).tolist()
            
        print(f"qa: {accuracy_score(answer['qa'], pred['qa'])}")
        writer.add_scalar("train", accuracy_score(answer['qa'], pred['qa']), ckpt)

if __name__ == "__main__":
    qa_train()
    exp_path = os.path.join("exp", "_qa", "_1")
    qa_test(exp_path, 'train')
    qa_test(exp_path, 'val')
