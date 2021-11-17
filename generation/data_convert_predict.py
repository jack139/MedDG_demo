import pickle
import copy
import os
import re

#with open('./data/160_last_topic2num.pk', 'rb') as f:
#    topic2num = pickle.load(f)

topic2num = {
    "胃炎": 0,
    "便秘": 1,
    "肠炎": 2,
    "感冒": 3,
    "食管炎": 4,
    "胃溃疡": 5,
    "胆囊炎": 6,
    "阑尾炎": 7,
    "肠易激综合征": 8,
    "胰腺炎": 9,
    "肝硬化": 10,
    "肺炎": 11,
    "排气": 12,
    "呕血": 13,
    "鼻塞": 14,
    "粘便": 15,
    "焦躁": 16,
    "头晕": 17,
    "菌群失调": 18,
    "呕吐": 19,
    "肛周疼痛": 20,
    "乏力": 21,
    "口苦": 22,
    "四肢麻木": 23,
    "食欲不振": 24,
    "胸痛": 25,
    "螺旋杆菌感染": 26,
    "黑便": 27,
    "尿频": 28,
    "痔疮": 29,
    "淋巴结肿大": 30,
    "咽部痛": 31,
    "水肿": 32,
    "精神不振": 33,
    "咳嗽": 34,
    "有痰": 35,
    "心悸": 36,
    "胃肠功能紊乱": 37,
    "反流": 38,
    "打嗝": 39,
    "过敏": 40,
    "肌肉酸痛": 41,
    "气促": 42,
    "发热": 43,
    "黄疸": 44,
    "月经紊乱": 45,
    "胃肠不适": 46,
    "恶心": 47,
    "腹泻": 48,
    "痉挛": 49,
    "肠梗阻": 50,
    "烧心": 51,
    "胃痛": 52,
    "便血": 53,
    "贫血": 54,
    "肠鸣": 55,
    "腹胀": 56,
    "头痛": 57,
    "尿急": 58,
    "里急后重": 59,
    "饥饿感": 60,
    "咽部灼烧感": 61,
    "稀便": 62,
    "腹痛": 63,
    "体重下降": 64,
    "脱水": 65,
    "呼吸困难": 66,
    "细菌感染": 67,
    "寒战": 68,
    "背痛": 69,
    "消化不良": 70,
    "喷嚏": 71,
    "吞咽困难": 72,
    "嗜睡": 73,
    "位置": 74,
    "诱因": 75,
    "性质": 76,
    "时长": 77,
    "胃蛋白酶": 78,
    "肛门镜": 79,
    "肝胆胰脾超声": 80,
    "尿检": 81,
    "CT": 82,
    "胶囊内镜": 83,
    "糖尿病": 84,
    "胃镜": 85,
    "B超": 86,
    "腹腔镜": 87,
    "结肠镜": 88,
    "尿常规": 89,
    "转氨酶": 90,
    "肠镜": 91,
    "呼气实验": 92,
    "血常规": 93,
    "便常规": 94,
    "小肠镜": 95,
    "钡餐": 96,
    "腹部彩超": 97,
    "左氧氟沙星": 98,
    "健脾丸": 99,
    "耐信": 100,
    "谷氨酰胺肠溶胶囊": 101,
    "硫糖铝": 102,
    "藿香正气丸": 103,
    "乳酸菌素": 104,
    "胃复安": 105,
    "莫沙必利": 106,
    "马来酸曲美布汀": 107,
    "复方消化酶": 108,
    "香砂养胃丸": 109,
    "多潘立酮": 110,
    "四磨汤": 111,
    "泌特": 112,
    "整肠生": 113,
    "嗜酸乳杆菌": 114,
    "思连康": 115,
    "阿莫西林": 116,
    "开塞露": 117,
    "思密达": 118,
    "金双歧": 119,
    "胃苏": 120,
    "兰索拉唑": 121,
    "消炎利胆片": 122,
    "蒙脱石散": 123,
    "得舒特": 124,
    "三九胃泰": 125,
    "瑞巴派特": 126,
    "乳果糖": 127,
    "布洛芬": 128,
    "颠茄片": 129,
    "健胃消食片": 130,
    "康复新液": 131,
    "人参健脾丸": 132,
    "雷呗": 133,
    "肠溶胶囊": 134,
    "吗丁啉": 135,
    "果胶铋": 136,
    "马来酸曲美布丁": 137,
    "多酶片": 138,
    "诺氟沙星": 139,
    "诺氟沙星胶囊": 140,
    "抗生素": 141,
    "甲硝唑": 142,
    "胶体果胶铋": 143,
    "达喜": 144,
    "肠胃康": 145,
    "铝碳酸镁": 146,
    "曲美布汀": 147,
    "益生菌": 148,
    "补脾益肠丸": 149,
    "奥美": 150,
    "克拉霉素": 151,
    "马来酸曲美布汀片": 152,
    "培菲康": 153,
    "山莨菪碱": 154,
    "磷酸铝": 155,
    "莫沙比利": 156,
    "吗叮咛": 157,
    "斯达舒": 158,
    "泮托拉唑": 159
}


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

