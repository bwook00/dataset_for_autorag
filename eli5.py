import os
import pathlib
from datetime import datetime

import pandas as pd
from datasets import load_dataset


def eli5():
    file_path = "NomaDamas/eli5"
    qa_data = load_dataset(file_path + "-qa")['train'].to_pandas()
    corpus_data = load_dataset(file_path + "-document")['train'].to_pandas()

    qa_data = qa_data.dropna()

    qa_data = qa_data[:1000]

    # make real corpus data
    doc_id_list = qa_data['doc_id'].tolist()

    real_corpus_data = pd.concat([
        corpus_data[corpus_data['doc_id'].isin(doc_id_list)],
        corpus_data[~corpus_data['doc_id'].isin(doc_id_list)][:1000]
    ], ignore_index=True)

    qa_data.rename(columns={'query_id': 'qid', 'question': 'query',
                            'goldenAnswer': 'generation_gt', 'doc_id': 'retrieval_gt'}, inplace=True)

    # Using apply with a lambda function to avoid explicit for-loops for restructuring.
    qa_data['retrieval_gt'] = qa_data['retrieval_gt'].apply(lambda x: [[x]])
    qa_data['generation_gt'] = qa_data['generation_gt'].apply(lambda x: [x])

    real_corpus_data = real_corpus_data.rename(columns={'document': 'contents'})

    real_corpus_data = real_corpus_data.drop(columns=['id'])

    metadata_dict = {'last_modified_datetime': datetime.now()}
    real_corpus_data['metadata'] = [metadata_dict for _ in range(len(real_corpus_data))]

    assert len(qa_data) == 1000
    assert len(real_corpus_data) == 2000

    # path setting
    root_dir = pathlib.PurePath(__file__).parent
    project_dir = os.path.join(root_dir, "eli5_project")

    # save qa data and corpus data
    qa_data.to_parquet(os.path.join(project_dir, "qa.parquet"), index=False)
    real_corpus_data.to_parquet(os.path.join(project_dir, "corpus.parquet"), index=False)


if __name__ == '__main__':
    eli5()
