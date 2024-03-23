# UitFedSecAggre
## Cài đặt file trước khi chạy
### File config_training.json

Sửa giá trị của session (chuỗi, tên, id, v.v.. gì tùy), để mỗi khi thực hiện chạy thì sẽ tạo ra 1 folder lưu lại kết quả lần train đó. Mỗi lần chạy thì nên sửa lại trường session này để tạo thêm folder mới tương ứng với lần chạy mới 

### File api_key.json

Trong thư mục library, copy tạo 1 file tên api_key.json từ file template, sau đó bỏ key và secret vào. Đây là API key dành cho IPFS Pinata.

### Solidity

+ Chỉnh sửa lại kết nối, port trong truffle-config.js. Cho đúng với setting trong Ganache. Có thể dùng notebook interact.ipynb trong folder server để test kết nối. 
+ Vào ganache lấy 2 địa chỉ từ 2 account bất kỳ và bỏ lần lượt vào file client1.json và client2.json. Để các client này được nhận reward

### SSL 

Bật terminal lên, cd đến thư mục certificates. chạy lệnh ./generate.sh

<span style="color: red;">Lưu ý</span>

Nhớ set lại các đường dẫn data trong file config_training.json

## Chạy 1 quá trình train

Bật 3 terminal, 2 client thì lần lượt chạy các file như client_api.py, client_api2.py. Chạy server thì chạy file server_api.py 

Muốn tăng số client chạy lên thì chỉnh sửa file config, copy file client_api.py và thay tham số trong hàm laucnh() 
