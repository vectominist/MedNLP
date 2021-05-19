import jieba
import unicodedata
import re
import string

chinese_punctuations = '，。、！？'
remove_punctuations = string.punctuation + chinese_punctuations
translator = str.maketrans(
    remove_punctuations, ' ' * len(remove_punctuations))

jieba.suggest_freq(('護理師'), True)
jieba.suggest_freq(('醫師'), True)
jieba.suggest_freq(('個管師'), True)
jieba.suggest_freq(('家屬'), True)
jieba.suggest_freq(('民眾'), True)


def split_sent(sentence: str):
    first_role_idx = re.search(':', sentence).end(0)
    out = [sentence[:first_role_idx]]

    tmp = sentence[first_role_idx:]
    while tmp:
        res = re.search(
            r'(護理師[\w*]\s*:|醫師\s*:|民眾\s*:|家屬[\w*]\s*:|個管師\s*:)', tmp)
        if res is None:
            break

        idx = res.start(0)
        idx_end = res.end(0)
        out[-1] = list(out[-1] + tmp[:idx])
        out.append(tmp[idx:idx_end])
        tmp = tmp[idx_end:]

    out[-1] = list(out[-1] + tmp)

    return out


def normalize_sent_with_jieba(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace('.', '')
    text = text.replace(' ', '')
    text = re.sub(r'\([^)]*\)', '', text)
    text = ["".join(i) for i in split_sent(text)]
    text = [t.translate(translator) for t in text]
    text = [(','.join(jieba.cut(t))).split(',') for t in text]
    for i in range(len(text)):
        text[i] = [w for w in text[i] if w != ' ']

    return text


if __name__ == '__main__':
    text = r"""個管師：好喔，我剛剛後來確認了一下你抽血的狀況（好的），哈哈～對，你下一次抽血就10月20到11月1號。民眾：好。個管師：然後廖醫師有幫你排回診2月17號。民眾：好。個管師：這樣ok嗎？民眾：好。個管師：然後你現階段……民眾：是。個管師：你跟你男朋友都沒有約嗎？民眾：沒有啊。個管師：然後這一個月沒有性行為？民眾：也沒有阿。個管師：你們沒有住在一起對不對？民眾：有。個管師：你們有住在一起，你們有住在一起，但是性行為可以預期嗎？民眾：蛤，可以啊。個管師：呵～真的喔？民眾：一個，因為我看，我是看著一個就是現實社會的狀況比那個更重的人，所以……個管師：什麼意思？民眾：就是，因為他現在都是工讀阿，所以他現在打了兩份工，所以那個時間，然後所以時間到我一定逼他去工作，呵呵～個管師：喔他現在在，誒他現在在工作了？民眾：他現在是在工作啦。個管師：他剛畢業嗎？民眾：沒有阿。個管師：畢業兩年？民眾：一段時間了。個管師：一年？民眾：已經超過了，對然後……個管師：那位什麼還在考license？"""
    print(normalize_sent_with_jieba(text))
