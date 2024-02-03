import os
import pathlib

from datetime import datetime
import pandas as pd
from pandas import json_normalize

from datasets import load_dataset


def msmarco():
    file_path = "ms_marco"

    dataset = load_dataset(file_path, 'v1.1')['test'].to_pandas()

    qid = dataset['query_id'].tolist()[:1000]
    query = dataset['query'].tolist()[:1000]
    generation_gt = dataset['answers'].tolist()[:1000]

    make_result = pd.concat(
        [dataset['query_id'], json_normalize(dataset['passages'])],
        axis=1
    )

    result = make_result.apply(__make_passages_and_retrieval_gt, axis=1)
    doc_id, contents, retrieval_gt = zip(*result)

    qa_data = pd.DataFrame({'qid': qid,
                            'query': query,
                            'generation_gt': generation_gt,
                            'retrieval_gt': retrieval_gt[:1000]
                            })

    doc_id_list = list(doc_id)
    contents_list = list(contents)
    flattened_doc_id = [item for sublist in doc_id_list for item in sublist]
    flattened_contents = [item for sublist in contents_list for item in sublist]

    corpus_data = pd.DataFrame({'doc_id': flattened_doc_id,
                                'contents': flattened_contents
                                })

    retrieval_gt_list = qa_data['retrieval_gt'].tolist()
    flattened_retrieval_gt = [item for sublist in retrieval_gt_list for item in sublist]

    real_corpus_data = pd.concat([
        corpus_data[corpus_data['doc_id'].isin(flattened_retrieval_gt)],
        corpus_data[~corpus_data['doc_id'].isin(flattened_retrieval_gt)][:1000]
    ], ignore_index=True)

    metadata_dict = {'last_modified_datetime': datetime.now()}
    real_corpus_data['metadata'] = [metadata_dict for _ in range(len(real_corpus_data))]

    # path setting
    root_dir = pathlib.PurePath(__file__).parent
    project_dir = os.path.join(root_dir, "msmarco_project")

    # save qa data and corpus data
    qa_data.to_parquet(os.path.join(project_dir, "msmarco_qa.parquet"), index=False)
    real_corpus_data.to_parquet(os.path.join(project_dir, "msmarco_corpus.parquet"), index=False)


def __make_passages_and_retrieval_gt(row):
    retrieval_gt = []
    doc_id, contents = [], []
    for passage_idx, passage_text in enumerate(row['passage_text']):
        doc_id.append(str(row['query_id']) + '_' + str(passage_idx))
        contents.append(passage_text)

        # Make retrieval gt and retrieval gt order.(27 is max count of passage texts in v2.1)
        if row['is_selected'][passage_idx] == 1:
            retrieval_gt.append(str(row['query_id']) + '_' + str(passage_idx))

    return doc_id, contents, retrieval_gt


if __name__ == '__main__':
    msmarco()
