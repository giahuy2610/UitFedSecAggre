import json
import pinatapy
import requests
import typing as tp
import os
from read_file import read_api_key
def upload_file_to_ipfs(file_path,dest_folder_name):
    pinata_api_key = None
    pinata_secret_api_key = None
    with open('api_key.json') as f:
        data = json.load(f)
        pinata_api_key = data['pinata']
        pinata_secret_api_key = pinata_api_key['secret']
        pinata_api_key = pinata_api_key['key']
    pinata = pinatapy.PinataPy(pinata_api_key, pinata_secret_api_key)
    response=pinata.pin_file_to_ipfs(file_path,ipfs_destination_path=dest_folder_name+'/',save_absolute_paths=False)
    return response

#source: https://stackoverflow.com/questions/74350228/how-do-i-upload-a-folder-containing-metadata-to-pinata-using-a-script-in-python
#Parameters: 
#folder_path and directory before it
#Ex: folder_path = 'server_result/1234567890'
# => penultimate_file = 'server_result'
def upload_folder_to_pinata(folder_path,penultimate_file):
    all_files: tp.List[str] = get_all_files(folder_path)
    key,secret=read_api_key('pinata')
    headers = {
        "pinata_api_key": key,
        "pinata_secret_api_key": secret,
    }
    # The replace function is a must, 
    # pinata servers doesn't recognize the backslash. 
    # Your filepath is probably different than mine,
    # so in the split function put your "penultimate_file/".
    # Strip the square brackets and the apostrophe,
    # because we don't want it as part of the metadata ipfs name
    files = [
        (
            "file",
            (
                str(file.replace("\\", "/").split(penultimate_file+"/")[-1:])
                .strip("[]")
                .strip("'"),
                open(file, "rb"),
            ),
        )
        for file in all_files
    ]
    response: requests.Response = requests.post(
        "https://api.pinata.cloud/pinning/pinFileToIPFS",
        files=files,
        headers=headers,
    )
    # If you want to see all the stats then do this: 
    # return/print/do both separately response.json()
    return response.json()

def get_all_files(directory: str) -> tp.List[str]:
    """get a list of absolute paths to every file located in the directory"""
    paths: tp.List[str] = []
    for root, dirs, files_ in os.walk(os.path.abspath(directory)):        
        for file in files_:            
             paths.append(os.path.join(root, file))
    return paths