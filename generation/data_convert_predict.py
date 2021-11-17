import pickle
import copy
import os
import re

with open('./data/160_last_topic2num.pk', 'rb') as f:
    topic2num = pickle.load(f)



def add_split_token(sentence):
    
    if sentence is not None and len(sentence) > 0 and re.search(u'[\u4e00-\u9fa5]+', sentence[-1]):
        sentence = sentence + "。"
    
    return sentence

def get_e(sen):
    
    return (
        sorted(set(sen['Symptom'])) 
        + sorted(set(sen['Attribute'])) 
        + sorted(set(sen['Test'])) 
        + sorted(set(sen['Disease'])) 
        + sorted(set(sen['Medicine']))
    )

def convert(test_add_label_entities_with_predict_entities):
    dig_test_data = []

    for dialog in test_add_label_entities_with_predict_entities:
        new_dialog = []
        history = []

        for sen in dialog:

            sen['Sentence'] = add_split_token(sen['Sentence'])

            aa = get_e(sen)

            aa = [item for item in aa if item in topic2num]

            if 'bert_word' in sen:
                aa = sen['bert_word']

            if len(history) > 0 and sen['id'] == 'Doctor':
                new_dialog.append({
                    "history": copy.deepcopy(history),
                    "bert_word": copy.deepcopy(aa),
                    'response': sen['Sentence'],
                })

            history.append(("病人：" if sen['id'] == "Patients" else "医生：") + sen['Sentence'])
            
        for dic in new_dialog[-1:]:

            dic['history'][-1] = dic['history'][-1] + "<" + ','.join(dic['bert_word']) + ">"

            dig_test_data.append(dic)

    return dig_test_data


if __name__ == '__main__':
    # 单独处理测试集
    with open('./data/test_add_info_entities_with_predict_entities.pk', 'rb') as f:
        test_data = pickle.load(f)

    dig_data = convert(test_data)

    with open('./data/T5_dig_test_data.pk','wb') as f:
        pickle.dump(dig_data, f)

