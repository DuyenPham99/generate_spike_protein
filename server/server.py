from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin

# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/add', methods=['POST', 'GET'])
@cross_origin(origin='*')
def add_process():
    a = int(request.args.get('sothunhat'))
    b = int(request.args.get('sothuhai'))
    kq = a + b
    return 'Kết quả là: ' + str(kq)


# Start Backend


if __name__ == '__main__':
    app.run(host='192.168.1.157', port='9999')
