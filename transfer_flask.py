import flask
import threading

falsk_app = flask.Flask(__name__)  



# 全局变量存储数据  
global_data = {} 



@falsk_app.route('/store', methods=['POST'])  
def store_data():  
    """存储数据到全局变量"""  
    data = flask.request.json  
    global_data.update(data)  
    return flask.jsonify({"status": "success", "message": "Data stored successfully"})  

@falsk_app.route('/fetch', methods=['GET'])  
def fetch_data():  
    """从全局变量获取数据"""  
    print(f'someone request global_data{global_data}')
    return flask.jsonify(global_data)  

def run_flask():  
    falsk_app.run(host='0.0.0.0', port=5001)  

if __name__ == "__main__":
    run_flask()