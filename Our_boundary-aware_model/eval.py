# coding: utf-8

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from dataset import End2EndDataset
from utils.torch_util import calc_f1
from utils.path_util import from_project_root


def evaluate_e2e(model, data_url, bsl_model=None):
    """ evaluating end2end model on dataurl

    Args:
        model: trained end2end model
        data_url: url to test dataset for evaluating
        bsl_model: trained binary sequence labeling model

    Returns:
        ret: dict of precision, recall, and f1

    """
    print("\nevaluating model on:", data_url, "\n")
    dataset = End2EndDataset(data_url, next(model.parameters()).device, evaluating=True)
    loader = DataLoader(dataset, batch_size=200, collate_fn=dataset.collate_func)
    ret = {'precision': 0, 'recall': 0, 'f1': 0}

    # switch to eval mode
    model.eval()
    with torch.no_grad():
        sentence_true_list, sentence_pred_list = list(), list()
        region_true_list, region_pred_list = list(), list()
        region_true_count, region_pred_count = 0, 0
        for data, sentence_labels, region_labels, records_list in loader:
            if bsl_model:
                pred_sentence_labels = torch.argmax(bsl_model.forward(*data), dim=1)
                pred_region_output, _ = model.forward(*data, pred_sentence_labels)
            else:
                try:
                    pred_region_output, pred_sentence_output = model.forward(*data)
                    # pred_sentence_output (batch_size, n_classes, lengths[0])
                    pred_sentence_labels = torch.argmax(pred_sentence_output, dim=1)
                    # pred_sentence_labels (batch_size, max_len)
                except RuntimeError:
                    print("all 0 tags, no evaluating this epoch")
                    continue

            # pred_region_output (n_regions, n_tags)
            pred_region_labels = torch.argmax(pred_region_output, dim=1).view(-1).cpu()
            # (n_regions)

            lengths = data[1]
            ind = 0
            for sent_labels, length, true_records in zip(pred_sentence_labels, lengths, records_list):
                pred_records = dict()
                for start in range(0, length):
                    if sent_labels[start] == 1:
                        if pred_region_labels[ind]>0: 
                            pred_records[(start,start+1)] = pred_region_labels[ind]
                        ind += 1
                        for end in range(start + 1, length):
                            if sent_labels[end] == 2:
                                if pred_region_labels[ind]:
                                    pred_records[(start,end+1)] = pred_region_labels[ind]
                                ind += 1

                for region in true_records:
                    true_label = dataset.label_ids[true_records[region]]
                    pred_label = pred_records[region] if region in pred_records else 0
                    region_true_list.append(true_label)
                    region_pred_list.append(pred_label)
                for region in pred_records:
                    if region not in true_records:
                        region_pred_list.append(pred_records[region])
                        region_true_list.append(0)

            region_labels = region_labels.view(-1).cpu()
            region_true_count += int((region_labels > 0).sum())
            region_pred_count += int((pred_region_labels > 0).sum())

            pred_sentence_labels = pred_sentence_labels.view(-1).cpu()
            sentence_labels = sentence_labels.view(-1).cpu()
            for tv, pv, in zip(sentence_labels, pred_sentence_labels):
                sentence_true_list.append(tv)
                sentence_pred_list.append(pv)

        print("sentence head and tail labeling result:")
        print(classification_report(sentence_true_list, sentence_pred_list,
                                    target_names=['out-entity', 'head-entity', 'tail-entity','in-entity'], digits=6))

        print("region classification result:")
        print(classification_report(region_true_list, region_pred_list,
                                    target_names=list(dataset.label_ids)[:12], digits=6))
        ret = dict()
        tp = 0
        for pv, tv in zip(region_pred_list, region_true_list):
            if pv == tv == 0:
                continue
            if pv == tv:
                tp += 1
        fp = region_pred_count - tp
        fn = region_true_count - tp
        ret['precision'], ret['recall'], ret['f1'] = calc_f1(tp, fp, fn)

    return ret


def main():
    model_url = from_project_root("data/model/best_model.pt")
    test_url = from_project_root("data/Germ/germ.test.iob2")
    model = torch.load(model_url)
    evaluate_e2e(model, test_url)
    pass


if __name__ == '__main__':
    main()
