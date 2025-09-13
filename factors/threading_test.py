import threading  
import time  
import random  

# 定义一个共享数据的列表  
shared_data = []  

# 定义一个事件，用于线程之间的同步  
data_ready_event = threading.Event()  

def producer():  
    """生产者线程，生成数据并放入共享列表"""  
    global shared_data  
    for _ in range(5):  
        # 生成随机数据  
        data = random.randint(1, 100)  
        shared_data.append(data)  
        print(f"生产者生成数据: {data}")  
        
        # 通知消费者数据已准备好  
        data_ready_event.set()  
        
        # 清除事件状态，准备下一次生产  
        time.sleep(random.uniform(0.5, 1.5))  # 模拟生产时间  
        data_ready_event.clear()  

def consumer():  
    """消费者线程，消费共享列表中的数据"""  
    global shared_data  
    for _ in range(5):  
        # 等待数据准备好  
        data_ready_event.wait()  
        
        # 消费数据  
        if shared_data:  
            data = shared_data.pop(0)  
            print(f"消费者消费数据: {data}")  
        
        # 模拟消费时间  
        time.sleep(random.uniform(0.5, 1.5))  

if __name__ == "__main__":  
    # 创建线程  
    producer_thread = threading.Thread(target=producer)  
    consumer_thread = threading.Thread(target=consumer)  

    # 启动线程  
    producer_thread.start()  
    consumer_thread.start()  

    # 等待线程完成  
    # producer_thread.join()  
    # consumer_thread.join()  

    # while(True):
    #     time.sleep(1)
    print("所有线程执行完毕。")