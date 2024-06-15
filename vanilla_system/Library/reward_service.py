from web3 import Web3, exceptions
import json
from typing import List

host = 'HTTP://127.0.0.1:7545'
web3 = Web3(Web3.HTTPProvider(host))

reward_contract=None
#load contract abi
with open('../Solidity/build/contracts/Reward.json') as f:
    compileJSON = json.load(f)
    abi = compileJSON['abi']
    contract_address= compileJSON['networks']['5777']['address']
    reward_contract = web3.eth.contract(address=contract_address, abi=abi)

class RewardService:
    def pay(self, address: str, amount: int):
        if web3.is_address(address):
            reward_contract.functions.addBalance(address, amount).transact({'from': web3.eth.accounts[0]})
        else:
            #raise exception InvalidAddress and print address
            raise exceptions.InvalidAddress(address)
            
    def getBalance(self, address: str):
        if web3.is_address(address):
            return reward_contract.functions.balanceOf(address).call()