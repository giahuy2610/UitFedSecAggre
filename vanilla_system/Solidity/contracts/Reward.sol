// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract Reward {
    string private _symbol = "FMT";
    string private _name = "Federated";

    address private _owner;
    uint256 private _totalSupply;
    mapping(address => uint256) private _balances;

    struct User {
        address userAddress;
        uint256 balance;
    }

    constructor() {
        _owner = msg.sender;
    }

    function name() public view returns (string memory) {
        return _name;
    }

    function symbol() public view returns (string memory) {
        return _symbol;
    }

    //Get balance of an address
    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    //add balance to an address
    function addBalance(address account, uint256 amount) public {
        _balances[account] += amount;
    }
}
