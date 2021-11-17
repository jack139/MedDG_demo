import json
#import datetime,time,
import glob
import os
#import shutil
import pickle
from bert4keras.tokenizers import Tokenizer, SpTokenizer
#from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.models import build_transformer_model
from bert4keras.backend import keras, K #, batch_gather
#from bert4keras.optimizers import Adam
#from bert4keras.optimizers import extend_with_weight_decay
#from bert4keras.optimizers import extend_with_layer_adaptation
#from bert4keras.optimizers import extend_with_piecewise_linear_lr
#from bert4keras.optimizers import extend_with_gradient_accumulation
#from bert4keras.layers import Loss
from keras.layers import Lambda, Dense, Input #, Permute, Activation
from keras.models import Model
import numpy as np
from tqdm import tqdm
import tensorflow as tf

# 建立默认session
graph = tf.Graph()  # 解决多线程不同模型时，keras或tensorflow冲突的问题
session = tf.Session(graph=graph)


max_in_len = 512
max_out_len = 128

config_path = '../nlp_model/mt5_base/mt5_base_config.json'
pretrain_checkpoint_path = '../nlp_model/mt5_base/model.ckpt-1000000'
spm_path = '../nlp_model/mt5_base/sentencepiece_cn.model'
keep_tokens_path = '../nlp_model/mt5_base/sentencepiece_cn_keep_tokens.json'

tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>')
keep_tokens = json.load(open(keep_tokens_path))


# 使用4个模型
#weights_paths = glob.glob('../source/alala_meddg/param/outputModelWeights/ICLR_2021_Workshop_MLPCP_Track_1_生成*/best_weights')
# 只使用difficult模型（对内存小的环境）
weights_paths = glob.glob('../source/alala_meddg/param/outputModelWeights/ICLR_2021_Workshop_MLPCP_Track_1_生成_difficult*/best_weights')
# 只使用base模型（对内存小的环境）
#weights_paths = glob.glob('../source/alala_meddg/param/outputModelWeights/ICLR_2021_Workshop_MLPCP_Track_1_生成_base*/best_weights')

encoders = []
decoders = []
models = []

with graph.as_default():
    with session.as_default():
        for idx, name in enumerate(tqdm(weights_paths)):
            print('load: ', name)

            t5 = build_transformer_model(
                config_path=config_path,
                # checkpoint_path=pretrain_checkpoint_path,
                keep_tokens=keep_tokens,
                model='t5.1.1',
                return_keras_model=False,
                name='T5',
            )

            t5.model.load_weights(name)

            encoders.append(t5.encoder)
            decoders.append(t5.decoder)
            models.append(t5.model)


from bert4keras.snippets import AutoRegressiveDecoder

class AutoTitleMult(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):

        all_probas = []

        for idx, c_encoded in enumerate(inputs):
          
            with graph.as_default(): # 解决多线程不同模型时，keras或tensorflow冲突的问题
                with session.as_default():
                    probas = self.last_token(decoders[idx]).predict([c_encoded, output_ids])
              
            # 特殊情况移除过短序列的结束标记概率
            if output_ids.shape[1] < self.minlen - 1:
                probas[:,self.end_id] = 1e-12

            all_probas.append(probas)

        all_probas = np.array(all_probas)

        return all_probas.max(axis=0)

    def generate(self, c_tokens, topk=1):
        c_token_ids = tokenizer.tokens_to_ids(c_tokens)
        
        with graph.as_default(): # 解决多线程不同模型时，keras或tensorflow冲突的问题
            with session.as_default():
                c_encodeds = [encoder.predict(np.array([c_token_ids]))[0] for encoder in encoders]

        output_ids = self.beam_search(c_encodeds, topk=topk)  # 基于beam search
        
        return tokenizer.decode([int(i) for i in output_ids])

# 5个字符 3个特殊token
autotitle = AutoTitleMult(start_id=0, end_id=tokenizer._token_end_id, maxlen=max_out_len, minlen=5+3)

def get_c(data_item):
    
    c_tokens = []

    for item in data_item['history'][::-1]:
        item = tokenizer._tokenize(item)

        if len(c_tokens) + len(item) > max_in_len - 1:
            break

        c_tokens = item + c_tokens


    c_tokens = c_tokens + ["</s>"]

#     print(" ".join(c_tokens))
#     print("--------------------------")

    return c_tokens


def after_to_text(text):
    
    # 调整长度
    if len(text) <= 4:
        text = text + text[-1]*(5-len(text))
        
    text = text.replace("?", "？").replace(",", "，")
    
    return text

def predict(data):
    pred_replies = []
    for idx, data_item in enumerate(tqdm(data)):
        c_tokens = get_c(data_item)
        
        reply = after_to_text(autotitle.generate(c_tokens, topk=1))
        
        pred_replies.append(reply)
    
    return pred_replies

if __name__ == '__main__':
    # 开始验证
    # 加载数据

    data_path = "./data/"

    with open(os.path.join(data_path, "T5_dig_test_data.pk"), "rb") as f:
        dig_test_data = pickle.load(f)

    ans_max_confidence = predict(dig_test_data)
    print(ans_max_confidence)

    with open(os.path.join(data_path, "ans_max_confidence.pk"), 'wb') as f:
        pickle.dump(ans_max_confidence, f)
