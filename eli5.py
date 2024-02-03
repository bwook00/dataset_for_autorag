import os
import pathlib

import pandas as pd
from datasets import load_dataset


def eli5():
    file_path = "NomaDamas/eli5"
    qa_data = load_dataset(file_path + "-qa")['train'].to_pandas()
    corpus_data = load_dataset(file_path + "-document")['train'].to_pandas()

    qa_data = qa_data[:1000]

    # make real corpus data
    doc_id_list = qa_data['doc_id'].tolist()

    real_corpus_data = pd.concat([
        corpus_data[corpus_data['doc_id'].isin(doc_id_list)],
        corpus_data[~corpus_data['doc_id'].isin(doc_id_list)][:1000]
    ], ignore_index=True)

    # column name change
    qa_data = qa_data.rename(columns={'query_id': 'qid', 'question': 'query',
                                      'goldenAnswer': 'generation_gt', 'doc_id': 'retrieval_gt'})
    real_corpus_data = real_corpus_data.rename(columns={'id': 'metadata', 'document': 'contents'})

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