'''
    model evaluation by F1-score
'''

try:
    from decode_prediction import BIO_decoder
except:
    from utils.decode_prediction import BIO_decoder

def dict2char(dic):
    out = []
    for key,value in dic.items():
        out.append(key+'='+str(value))
    return '|'.join(out)

def f1_score(sentences, targets, predictions, sentences1=None, f1_type='micro', eps=1e-9):

    '''
    params:
    -------
        sentences: list(list(char))
        targets: list(list(char))
        predictions: list(list(char))
        f1_type: "micro"|"macro"
    '''

    # check input
    if not sentences1:sentences1 = sentences
    for sentence, target, sentence1, prediction in zip(sentences, targets,
            sentences1, predictions):
        assert len(sentence) == len(target) and len(sentence1) == len(prediction), print('input error!\n', sentence, target, prediction)

    all_TP, all_FN, all_FP = 0, 0, 0
    macro_f1 = 0
    for sentence, sentence1, target, prediction in zip(sentences, sentences1, targets, predictions):
        
        true_entity = BIO_decoder(sentence, target)
        pred_entity = BIO_decoder(sentence1, prediction)

        true_entity = [dict2char(i) for i in true_entity]
        pred_entity = [dict2char(i) for i in pred_entity]

        TP = len(set(true_entity) & set(pred_entity))
        FN = len(true_entity) - TP
        FP = len(pred_entity) - TP

        # update macro-f1
        Precision = (TP + eps) / (TP + FP + eps)
        Recall = (TP + eps) / (TP + FN + eps)
        macro_f1 += 2 * Precision * Recall / (Precision + Recall + eps) 

        # update micro-f1
        all_TP += TP
        all_FN += FN
        all_FP += FP

    if f1_type == 'micro':
        all_Precision = all_TP / (all_TP + all_FP + eps)
        all_Recall = all_TP / (all_TP + all_FN + eps)
        return 2 * all_Precision * all_Recall / (all_Precision + all_Recall +
                eps)
    else:
        return macro_f1 / len(sentences)

def f1_score_from_path(sentences_path, targets_path, predictions_path, sentences1_path,
        **kwargs):

    def _read_data(path):
        data = open(path, 'r').read().split('\n')
        data = [i.split() for i in data]
        return data

    if not sentences1_path:sentences1_path = sentences_path
    sentences = _read_data(sentences_path)
    targets = _read_data(targets_path)
    sentences1 = _read_data(sentences1_path)
    predictions = _read_data(predictions_path)

    return f1_score(sentences, targets, predictions, sentences1, **kwargs)

if __name__ == '__main__':
    path1 = '../msra/train/sentences.txt'
    path2 = '../msra/train/tags.txt'
    sentences = open(path1, 'r').read().split('\n')
    targets = open(path2, 'r').read().split('\n')
    sentences, targets = [i.split() for i in sentences], [i.split() for i in targets]
    print('micro-f1:', f1_score(sentences, targets, targets, f1_type='micro'))
    print('macro-f1:', f1_score(sentences, targets, targets, f1_type='macro'))

    print('micro_f1:', f1_score_from_path(path1, path2, path2, path1, f1_type='micro'))
    print('macro_f1:', f1_score_from_path(path1, path2, path2, path1,f1_type='macro'))
