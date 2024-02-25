import os
import json
import numpy as np
from ipfs_handler import upload_folder_to_pinata

def make_dir_if_not_exists(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def save_weights(aggregated_weights, session_id):
    make_dir_if_not_exists('./server_result')

    session_folder =f'./server_result/{session_id}'
    make_dir_if_not_exists(session_folder)
    np.save(session_folder+'/weights.npy', aggregated_weights)

def write_json_result_for_server(results, session_id):
    make_dir_if_not_exists('./server_result')

    session_folder =f'./server_result/{session_id}'
    make_dir_if_not_exists(session_folder)
    with open(session_folder+'/result.json', 'w') as f:
        json.dump(results, f, indent=4)

    rs=upload_folder_to_pinata(session_folder,penultimate_file='server_result')
    print(rs)

def write_json_result_for_client(data, session_id, client_id,round):
    client_result = f'client_result_{client_id}'
    make_dir_if_not_exists(client_result)

    session_folder =client_result+f'/{session_id}'
    make_dir_if_not_exists(session_folder)
    with open(session_folder+'/round_'+str(round)+'.json', 'w') as f:
        json.dump(data, f, indent=4)

def save_config_file(json_path, session_id,dictionary):
    with open(json_path, 'r') as file:
        data = json.load(file)
        for i in dictionary:
            data[i] = dictionary[i]
    dest=f'./server_result/{session_id}/config_training.json'
    with open(dest, 'w') as f:
        json.dump(data, f, indent=4)