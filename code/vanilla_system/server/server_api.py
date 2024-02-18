# File contains all api methods for server

from fastapi import FastAPI
from server_api_class import ServerApi
     
server=FastAPI()
control = ServerApi()
control.launch_fl_session()

@server.post("/config")
async def config():
    return

@server.post("/launchFL")
async def launch_fl():
    return control.launch_fl_session()           