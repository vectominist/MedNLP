import jieba
import unicodedata
import re
import string

chinese_punctuations = r"""，。、！？‘’·⋯‧”“"""
english_punctuations = string.punctuation.replace(':', '')
remove_punctuations = english_punctuations + chinese_punctuations
translator = str.maketrans(remove_punctuations, ' ' * len(remove_punctuations))
remove_chars = '嗯恩啊嘛呃唉哎誒欸痾喔ㄟ哦阿齁啦嘿哼亨蛤吧嗎呵餒'
translator2 = str.maketrans(remove_chars, ' ' * len(remove_chars))

word_dict = [
    '醫師', '民眾', '戴套', '無套', '性行為', '炮友', '口交',
    '固炮', '頭暈', '約炮', '愛滋', '帶套', '開放式', '封閉式',
    '處方籤', '回診', '關係', '肺結核', '抑鬱', '任務型', '小時',
    '結核', '破功', '單核球', '就醫'
]

for w in word_dict:
    jieba.suggest_freq((w), True)


def deEmojify(text):
    regrex_pattern = re.compile(pattern="["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def split_sent(sentence: str, split_type: str = 'period'):
    first_role_idx = re.search(':', sentence).end(0)
    out = [sentence[:first_role_idx]]

    tmp = sentence[first_role_idx:]
    while tmp:
        if split_type == 'period':
            res = re.search(
                r'(護理師[\w*]\s*:|醫師\s*:|民眾\s*:|家屬[\w*]\s*:|個管師\s*:|，|。|？|！)', tmp)
        else:
            res = re.search(
                r'(護理師[\w*]\s*:|醫師\s*:|民眾\s*:|家屬[\w*]\s*:|個管師\s*:|,|。|\?|!|~|⋯)', tmp)
        if res is None:
            break

        idx = res.start(0)
        idx_end = res.end(0)
        out[-1] = list(out[-1] + tmp[:idx])
        out.append(tmp[idx:idx_end])
        tmp = tmp[idx_end:]

    out[-1] = list(out[-1] + tmp)

    return out


def normalize_sent_with_jieba(
        text: str, split: bool = True, reduce: bool = True,
        max_sent_len: int = 20, remove_short: bool = True,
        remove_id: bool = False, add_id: bool = False,
        split_type: str = 'period'):
    '''
        Text normalization with jieba
        Inputs:
            text: input document
            split: whether to split into sentences (if input is a dialogue)
            reduce: remove speaker name
            max_sent_len: maximum sentence length
            remove_id: remove speaker identity
            add_id: add speaker identity
        Output:
            list of str
    '''
    text = unicodedata.normalize("NFKC", text).lower()
    # print(text)
    text = deEmojify(text)
    text = text.replace('.', '')
    text = text.replace(' ', '')
    text = text.replace('、', '')
    text = text.replace('個管師', '醫師')
    text = text.replace('護理師', '醫師')
    text = text.replace('家屬', '民眾')
    text = text.replace('民眾a:', '民眾:')
    text = text.replace('民眾b:', '民眾:')
    text = text.replace('民眾1:', '民眾:')
    text = text.replace('民眾2:', '民眾:')
    text = text.replace('砲', '炮')
    text = text.replace('爲', '為')
    text = text.replace('羣', '群')
    text = text.replace('甚麼', '什麼')
    text = text.replace('位什麼', '為什麼')
    text = re.sub('(哈)+', '哈', text)
    text = re.sub('(ok)+', 'ok', text)
    text = re.sub('(no)+', 'no', text)
    text = re.sub(r'\([^)]*\)', '', text)
    if split:
        text = ["".join(i) for i in split_sent(text, split_type=split_type)]
    else:
        text = [text]
    if remove_id:
        text = [t.replace('民眾:', '') for t in text]
        text = [t.replace('醫師:', '') for t in text]
    elif add_id:
        _text = []
        for i, t in enumerate(text):
            if t[:3] != '民眾:' and t[:3] != '醫師:':
                t = _text[-1][:3] + t
            _text.append(t)
        text = _text
    text = [t.translate(translator) for t in text]
    text = [t.translate(translator2) for t in text]
    text = [(','.join(jieba.cut(t))).split(',') for t in text]
    for i in range(len(text)):
        text[i] = [w for w in text[i] if w != ' ']
    if reduce:
        text = [t[2:] for t in text if len(t) > 2]
    else:
        if remove_short:
            text = [t for t in text if len(t) > 2]
    if reduce:
        out_text = [(t if len(t) <= max_sent_len else t[-max_sent_len:])
                    for t in text]
    else:
        out_text = [(t if len(t) <= max_sent_len else t[:2] + t[-max_sent_len:])
                    for t in text]

    return out_text


if __name__ == '__main__':
    text = r"""個管師：好喔，我剛剛後來確認了一下你抽血的狀況（好的）。哈哈～對，你下一次抽血就10月20到11月1號。民眾：好。個管師：然後廖醫師有幫你排回診2月17號。民眾：好。個管師：這樣ok嗎？民眾：好。個管師：然後你現階段……民眾：是。個管師：你跟你男朋友都沒有約嗎？民眾：沒有啊。個管師：然後這一個月沒有性行為？民眾：也沒有阿。個管師：你們沒有住在一起對不對？民眾：有。個管師：你們有住在一起，你們有住在一起，但是性行為可以預期嗎？民眾：蛤，可以啊。個管師：呵～真的喔？民眾：一個，因為我看，我是看著一個就是現實社會的狀況比那個更重的人，所以……個管師：什麼意思？民眾：就是，因為他現在都是工讀阿，所以他現在打了兩份工，所以那個時間，然後所以時間到我一定逼他去工作，呵呵～個管師：喔他現在在，誒他現在在工作了？民眾：他現在是在工作啦。個管師：他剛畢業嗎？民眾：沒有阿。個管師：畢業兩年？民眾：一段時間了。個管師：一年？民眾：已經超過了，對然後……個管師：那位什麼還在考license？"""

    # out = normalize_sent_with_jieba(text)
    out = normalize_sent_with_jieba(
        text, reduce=False, max_sent_len=20, add_id=True)
    out = [' '.join(t) for t in out]
    for t in out:
        print(''.join(t.split(' ')))
