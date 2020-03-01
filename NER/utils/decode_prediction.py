
'''
    Decode char-level predictions to entity struct:
        Example: 
            -input : 我:O 爱:O 北:B-LOC 京:I-LOC 天:I-LOC 安:I-LOC 门:I-LOC
            -output : {'start':2, 'end':7, 'entity':'北京天安门', 'type':'person'}
'''

def BIO_decoder(sentence, prediction):
    
    # for data marked by BIO symbol

    # check input data
    assert len(sentence) == len(prediction), print('input error!')

    entitys = []
    start, n = 0, len(sentence)

    while start < n:
        if prediction[start][0] == 'B':
            entity_type = prediction[start].split('-')[-1]
            end = start + 1
            while end < n and prediction[end][0] == 'I':
                end += 1
            entitys.append({'start':start, 
                            'end':end,
                            'entity':''.join(sentence[start:end]), 
                            'type':entity_type})
            start = end
        else:
            start += 1
    
    return entitys

if __name__ == '__main__':
    path1 = '../msra/train/sentences.txt'
    path2 = '../msra/train/tags.txt'
    sentences = open(path1, 'r').read().split('\n')
    targets = open(path2, 'r').read().split('\n')
    for sentence, target in zip(sentences[:5], targets[:5]):
        sentence, target = sentence.split(), target.split()
        print([i+':'+j for i, j in zip(sentence, target)])
        print(BIO_decoder(sentence, target))
        print('\n')
