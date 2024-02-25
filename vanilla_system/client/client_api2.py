# File contains all api methods for client

from fastapi import FastAPI
from client_api_class import ClientApi

client=FastAPI()
control = ClientApi()
# import string
# import random

# def get_random_string(length):
#     # choose from all lowercase letter
#     letters = string.ascii_lowercase
#     result_str = ''.join(random.choice(letters) for i in range(length))
#     return result_str


control.launch_fl_session("2")

#   Participate in the fl session
@client.post("/client-participate")
async def participate(clientId: str):
    return control.launch_fl_session(clientId)

#   Get data analysis report

#   Get contribute analysis report

#   Get contributed history 

#   Get token balance
