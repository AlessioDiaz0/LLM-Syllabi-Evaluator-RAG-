import os
import json
import time

SAVING_DIR = 'documents/generated_output'

def save_to_json(result):
    data = get_data(result)

    if not os.path.exists(SAVING_DIR):
        os.makedirs(SAVING_DIR)
    
    # Add timestamp to filename
    file_name = time.strftime("evaluator_output_%H-%M-%S_-%m-%d-%Y.json")

    file_path = os.path.join(SAVING_DIR, file_name)
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)
    
    print(f"\nData has been saved to {file_path}")

def get_data(result):
    answer = result['answer_builder']['answers'][0]
    measured_latency = answer.meta['eval_count'] / answer.meta['eval_duration']

    data = {'model:': answer.meta['model'],
            'generated_answer': answer.data,
            'file(s)_used:': [doc.meta['file_path'] for doc in answer.documents],
            'score(s)': [doc.score for doc in answer.documents],
            'measured_latency': measured_latency,
            'vector(s)_size:': [len(doc.embedding) for doc in answer.documents],
            'done_reason': answer.meta['done_reason']
            }
    
    return data