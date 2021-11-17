import pickle
import json
from entity_recognition.model import predict as ner_predict
from next_entities.model import predict_test as next_entity_predict
from generation.data_convert_predict import convert
from generation.model import predict as output_predict

def prepare_data(test_data):
    test_add_info = []

    for data_item in test_data:
        
        test_add_info_item = []
        
        for s in data_item['history']:
            s_label = "Patient" if s[:3]=="患者：" else "Doctor"

            test_add_info_item.append(
                {
                    'id': s_label,
                    'Sentence': s[3:],
                    'Symptom': [],
                    'Medicine': [],
                    'Test': [],
                    'Attribute': [],
                    'Disease': []
                }
            )
        
        test_add_info.append(test_add_info_item)

    return test_add_info


def predict_flow(test_data):
    test_add_info = prepare_data(test_data)
    test_add_info_entities = ner_predict(test_add_info)
    test_next_entities = next_entity_predict(test_add_info_entities)
    t5_dig_data = convert(test_next_entities)
    t5_output = output_predict(t5_dig_data)

    return t5_output


if __name__ == '__main__':

    #test_path = "./data/test.json"
    #test_data = json.load(open(test_path))

    test_data = [
        {
            "history": [
                "患者：十二指肠溃疡的症状有哪些（女，17岁）",
                "医生：您好，有什么症状吗？",
                "患者：夜间会出现饥饿痛症状。",
                "患者：吃东西后会有缓解。",
                "医生：明白，这还真与十二指肠溃疡症状相似。",
                "患者：然后服用了两天奥美拉唑。",
                "患者：还是没有很大改善。",
                "患者：也不能说是疼吧，就是感觉很饿，但是距离餐后并没有很久。",
                "患者：一缩一缩的疼。",
                "医生：你有没有到医院检查一下？",
                "患者：只在夜间出现。",
                "患者：没有过。",
                #"医生：建议查一下幽门螺杆菌以及做一个胃镜看看。",
                #"患者：那冒昧问一下，这种症状出现会是什么很严重的疾病吗？",
                #"医生：根据你提供的症状，考虑存在消化性溃疡的可能，尤其你存在饥饿行疼痛，考虑还是十二指肠溃疡的可能性比较大。饥饿性。",
                #"患者：好的谢谢，那奥美拉唑需要停药吗？"
            ]
        }
    ]

    output = predict_flow(test_data)

    print(output)
