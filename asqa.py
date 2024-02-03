import os
import pathlib

from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd


from datasets import load_dataset


def asqa():
    file_path = "din0s/asqa"
    dataset = load_dataset(file_path)['dev'].to_pandas().reset_index(drop=True)

    dataset['retrieval_gt'], dataset['answer_gt'] = zip(*dataset.apply(__split_content_answer, axis=1))
    dataset = dataset.dropna(ignore_index=True)

    qid = dataset['sample_id'].tolist()
    query = dataset['ambiguous_question'].tolist()[:500]
    generation_gt = dataset['answer_gt'].tolist()[:500]

    information = dataset[['sample_id', 'retrieval_gt', 'wikipages']]
    sample_retrieval_gt = deepcopy(information)
    retrieval_gt = sample_retrieval_gt.apply(__make_retrieval_gt, axis=1).tolist()
    contents = information.apply(__make_contents, axis=1).tolist()
    doc_id = [query_id + '_' + str(i) for i, query_id in enumerate(qid)]

    qa_data = pd.DataFrame({'qid': qid[:500],
                            'query': query,
                            'generation_gt': generation_gt,
                            'retrieval_gt': retrieval_gt[:500]
                            })

    corpus_data = pd.DataFrame({'doc_id': doc_id,
                                'contents': contents
                                })

    metadata_dict = {'last_modified_datetime': datetime.now()}
    corpus_data['metadata'] = [metadata_dict for _ in range(len(corpus_data))]

    # path setting
    root_dir = pathlib.PurePath(__file__).parent
    project_dir = os.path.join(root_dir, "asqa_project")

    # save qa data and corpus data
    qa_data.to_parquet(os.path.join(project_dir, "asqa_qa.parquet"), index=False)
    corpus_data.to_parquet(os.path.join(project_dir, "asqa_corpus.parquet"), index=False)


def __split_content_answer(row):
    content_lst = []
    answer_lst = []
    for element in row['annotations']:
        if len(element['knowledge']) != 0:
            content_lst += [content for content in element['knowledge']]
        answer_lst += [element['long_answer']]

    if len(content_lst) == 0 or len(answer_lst) == 0:
        return np.NAN, np.NAN
    else:
        return content_lst, answer_lst


def __make_retrieval_gt(row):
    gt = [str(row['sample_id']) + '_' + str(idx) for idx, content in enumerate(row['retrieval_gt'])]
    return gt


def __make_contents(row):
    contents = []
    for content_dict in row['retrieval_gt']:
        contents.append(content_dict['content'])
    return contents


if __name__ == '__main__':
    asqa()
