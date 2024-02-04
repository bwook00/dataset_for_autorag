import os
import pathlib
import uuid

from datetime import datetime
import pandas as pd

from datasets import load_dataset


def dstc():
    file_path = "NomaDamas/DSTC-11-Track-5"

    qa = load_dataset(file_path, 'default')['test'].to_pandas().dropna()
    knowledge = load_dataset(file_path, 'knowledge')['train'].to_pandas()

    qid = [str(uuid.uuid4()) for _ in range(len(qa))][:1000]

    query, retrieval_gt, generation_gt = zip(*qa.apply(__preprocess_prompt, axis=1))
    knowledge['doc_id'] = knowledge.apply(__renewal_doc_id, axis=1)

    split_retrieval_gt = retrieval_gt[:1000]


    qa_data = pd.DataFrame({'qid': qid,
                            'query': query[:1000],
                            'generation_gt': generation_gt[:1000],
                            'retrieval_gt': split_retrieval_gt
                            })

    # Remove rows where 'list_column' is an empty list
    qa_data = qa_data[qa_data['generation_gt'].apply(lambda x: len(x) > 0)]
    qa_data = qa_data[qa_data['retrieval_gt'].apply(lambda x: len(x) > 0)]

    qa_data['generation_gt'] = qa_data['generation_gt'].apply(lambda x: [x])

    # [1,2], [3] -> [[1],[2]], [[3]]
    retrieval_gt_list = qa_data['retrieval_gt'].tolist()
    output_list = [[[item] for item in sublist] if isinstance(sublist, list) and all(
        isinstance(elem, str) and "_" in elem for elem in sublist) else sublist for sublist in retrieval_gt_list]
    qa_data['retrieval_gt'] = output_list

    doc_id, contents = zip(*knowledge.apply(__make_id_content, axis=1))

    corpus_data = pd.DataFrame({'doc_id': doc_id,
                                'contents': contents
                                })
    # [[1],[2]], [[3]] -> [1,2,3]
    flattened_retrieval_gt = [item for sublist in retrieval_gt_list for item in sublist]

    real_corpus_data = pd.concat([
        corpus_data[corpus_data['doc_id'].isin(flattened_retrieval_gt)],
        corpus_data[~corpus_data['doc_id'].isin(flattened_retrieval_gt)][:1000]
    ], ignore_index=True)

    metadata_dict = {'last_modified_datetime': datetime.now()}
    real_corpus_data['metadata'] = [metadata_dict for _ in range(len(real_corpus_data))]

    # path setting
    root_dir = pathlib.PurePath(__file__).parent
    project_dir = os.path.join(root_dir, "dstc_project")

    # save qa data and corpus data
    qa_data.to_parquet(os.path.join(project_dir, "qa.parquet"), index=False)
    real_corpus_data.to_parquet(os.path.join(project_dir, "corpus.parquet"), index=False)


def __preprocess_prompt(row):
    question = " ".join(
        [f"{prompt['speaker']}: {prompt['text']}" for prompt in row['log']])

    response = row['response']
    gt = []
    for knowledge in row['knowledge']:
        if knowledge['doc_type'] == 'review':
            gt.append("_".join(
                [str(knowledge['doc_id']), knowledge['doc_type'], knowledge['domain'],
                 str(knowledge['entity_id']), str(int(knowledge['sent_id']))]
            ))
        elif knowledge['doc_type'] == 'faq':
            gt.append("_".join(
                [str(knowledge['doc_id']), knowledge['doc_type'],
                 knowledge['domain'], str(knowledge['entity_id'])]
            ))

    return question, gt, response

def __make_id_content(row):
    if row['doc_type'] == 'review':
        content = row['review_sentence']
    elif row['doc_type'] == 'faq':
        content = row['faq_question'] + ', ' + row['faq_answer']

    return row['doc_id'], content


def __renewal_doc_id(row):
    if row['doc_type'] == 'review':
        return "_".join(
            [str(row['doc_id']), row['doc_type'], row['domain'],
             str(row['entity_id']), str(row['review_sent_id'])]
        )
    elif row['doc_type'] == 'faq':
        return "_".join(
            [str(row['doc_id']), row['doc_type'],
             row['domain'], str(row['entity_id'])]
        )


if __name__ == '__main__':
    dstc()
