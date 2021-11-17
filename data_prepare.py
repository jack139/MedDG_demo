import os
import pickle, json

all_test_path = "./data/test.json"
test_data_path = "./data/test_add_info.pk"


# 测试数据
test_data = json.load(open(all_test_path))

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

with open(test_data_path, "wb") as f:
    pickle.dump(test_add_info, f)

#json.dump(
#    test_add_info,
#    open(test_data_path+'.json', 'w', encoding='utf-8'),
#    indent=4,
#    ensure_ascii=False
#)

print("测试数据预处理完成", len(test_data))
