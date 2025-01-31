
def metrics(X, y_true, y_pred, data_manager):
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    hit_num = 0
    pred_num = 0
    true_num = 0

    correct_label_num = 0
    total_label_num = 0

    label_num = {}
    label_metrics = {}
    measuring_metrics = ["accuracy","precision","recall","f1"]
    # tensor向量不能直接索引，需要转成numpy
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    X = X.numpy()
    # decode_out = open('/Users/didi/Desktop/KYCode/EmerAT-NER/decode_out.txt', "w",encoding='utf8')
    # decode_out = open('E:\\ARearchCode\\EmerAT-NER\\decode_out.txt', "w",encoding='utf8')
    # decode_out = open('./decode_out.txt', "w", encoding='utf8')

    for i in range(len(y_true)):
        x = data_manager.tokenizer.convert_ids_to_tokens(X[i].tolist(), skip_special_tokens=True)

        y = [str(data_manager.id2label[val]) for val in y_true[i] if val != data_manager.label2id[data_manager.PADDING]]
        y_hat = [str(data_manager.id2label[val]) for val in y_pred[i] if
                 val != data_manager.label2id[data_manager.PADDING]]  # if val != 5
        # print(i, len(y), len(x), len(y_hat))
        # 输出预测结果
        # xx = data_manager.tokenizer.convert_ids_to_tokens(X[i].tolist(),skip_special_tokens=False)
        # info = ''
        # # x中不包含CLS和SEP
        # for j in range(1,len(y)-1):
        #     if y[j] != data_manager.label2id[data_manager.PADDING]:
        #         info += '{} {} {}\n'.format(xx[j],y[j],y_hat[j])
        # info += "\n"
        # decode_out.write(info)
        # decode_out.flush()

        correct_label_num += len([1 for a, b in zip(y, y_hat) if a == b])
        total_label_num += len(y)

        true_labels, labeled_labels_true, _ = extract_entity(x, y, data_manager)
        pred_labels, labeled_labels_pred, _ = extract_entity(x, y_hat, data_manager)

        hit_num += len(set(true_labels) & set(pred_labels))
        pred_num += len(set(pred_labels))
        true_num += len(set(true_labels))

        for label in data_manager.suffix:
            label_num.setdefault(label, {})
            label_num[label].setdefault('hit_num', 0)
            label_num[label].setdefault('pred_num', 0)
            label_num[label].setdefault('true_num', 0)

            true_lab = [x for (x, y) in zip(true_labels, labeled_labels_true) if y == label]
            pred_lab = [x for (x, y) in zip(pred_labels, labeled_labels_pred) if y == label]

            label_num[label]['hit_num'] += len(set(true_lab) & set(pred_lab))
            label_num[label]['pred_num'] += len(set(pred_lab))
            label_num[label]['true_num'] += len(set(true_lab))

    info = 'label_true:{},label_pred_true:{},label_true_true：{},'.format(true_num, pred_num, hit_num)
    print(info)
    if total_label_num != 0:
        accuracy = 1.0 * correct_label_num / total_label_num

    if pred_num != 0:
        precision = 1.0 * hit_num / pred_num
    if true_num != 0:
        recall = 1.0 * hit_num / true_num
    if precision > 0 and recall > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    # 按照字段切分
    for label in label_num.keys():
        tmp_precision = 0.0
        tmp_recall = 0.0
        tmp_f1 = 0.0
        # 只包括BI
        if label_num[label]['pred_num'] != 0:
            tmp_precision = 1.0 * label_num[label]['hit_num'] / label_num[label]['pred_num']
        if label_num[label]['true_num'] != 0:
            tmp_recall = 1.0 * label_num[label]['hit_num'] / label_num[label]['true_num']
        if tmp_precision > 0 and tmp_recall > 0:
            tmp_f1 = 2.0 * (tmp_precision * tmp_recall) / (tmp_precision + tmp_recall)
        label_metrics.setdefault(label, {})
        label_metrics[label]['precision'] = tmp_precision
        label_metrics[label]['recall'] = tmp_recall
        label_metrics[label]['f1'] = tmp_f1
    # decode_out.close()
    results = {}
    for measure in measuring_metrics:
        results[measure] = vars()[measure]
    return results, label_metrics


import re


def extract_entity_(sentence, labels_, reg_str, label_level):
    entices = []
    labeled_labels = []
    labeled_indices = []
    labels__ = [('%03d' % ind) + lb for lb, ind in zip(labels_, range(len(labels_)))]
    labels = ' '.join(labels__)

    re_entity = re.compile(reg_str)

    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        if label_level == 1:
            labeled_labels.append('_')
        elif label_level == 2:
            labeled_labels.append(entity_labels.split()[0][5:])

        start_index = int(entity_labels.split()[0][:3])
        if len(entity_labels.split()) != 1:
            end_index = int(entity_labels.split()[-1][:3]) + 1
        else:
            end_index = start_index + 1
        entity = ' '.join(sentence[start_index:end_index])
        labels = labels__[end_index:]
        labels = ' '.join(labels)
        entices.append(entity)
        labeled_indices.append((start_index, end_index))
        m = re_entity.search(labels)

    return entices, labeled_labels, labeled_indices


def extract_entity(x, y, data_manager):
    label_scheme = "BIO"
    label_level = 2
    label_hyphen = "-"
    reg_str = ''
    if label_scheme == 'BIO':
        if label_level == 1:
            reg_str = r'([0-9][0-9][0-9]B' + r' )([0-9][0-9][0-9]I' + r' )*'

        elif label_level == 2:
            tag_bodies = ['(' + tag + ')' for tag in data_manager.suffix]
            tag_str = '(' + ('|'.join(tag_bodies)) + ')'
            reg_str = r'([0-9][0-9][0-9]B' + label_hyphen + tag_str + r' )([0-9][0-9][0-9]I' + label_hyphen + tag_str + r'\s*)*'

    elif label_scheme == 'BIESO':
        if label_level == 1:
            reg_str = r'([0-9][0-9][0-9]B' + r' )([0-9][0-9][0-9]I' + r' )*([0-9][0-9][0-9]E' + r' )|([0-9][0-9][0-9]S' + r' )'

        elif label_level == 2:
            tag_bodies = ['(' + tag + ')' for tag in data_manager.suffix]
            tag_str = '(' + ('|'.join(tag_bodies)) + ')'
            reg_str = r'([0-9][0-9][0-9]B' + label_hyphen + tag_str + r' )([0-9][0-9][0-9]I' + label_hyphen + tag_str + r' )*([0-9][0-9][0-9]E' + label_hyphen + tag_str + r' )|([0-9][0-9][0-9]S' + label_hyphen + tag_str + r' )'

    return extract_entity_(x, y, reg_str, label_level)

def compute_f1():
    ans = 0
    with open("/Users/didi/Desktop/KYCode/EmerAT-NER/decode_out_1.txt","r",encoding="utf8") as f:
        for line in f:

            arr = line.strip().split()
            if arr == "":
                continue
            elif len(arr)==3 and arr[1] == arr[2] and arr[1]!="O":
                print(line)
                ans += 1
    print(ans)


if __name__ == '__main__':
    compute_f1()