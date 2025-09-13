
from typing import List, Dict, Tuple 
import datetime

class OrderFilter:  
    """  
    订单过滤器：提供挂单过滤功能，包括短时间挂单、异常挂单量、价格异常等。  
    """  

    @staticmethod  
    def filter_short_lived_orders(  
        cumulative_bids: List[Dict], cumulative_asks: List[Dict], min_duration: int = 2  
    ) -> Tuple[List[Dict], List[Dict]]:  
        """  
        过滤掉停留时间过短的挂单。  
        :param cumulative_bids: 买单列表，每个元素是 {"price": ..., "volume": ..., "time": ...}  
        :param cumulative_asks: 卖单列表，每个元素是 {"price": ..., "volume": ..., "time": ...}  
        :param min_duration: 最小停留时间（秒）  
        :return: 过滤后的买单和卖单列表  
        """  
        now = datetime.datetime.now()  

        def filter_orders(orders: List[Dict]) -> List[Dict]:  
            return [  
                order  
                for order in orders  
                if (now - order["time"]).total_seconds() >= min_duration  
            ]  

        filtered_bids = filter_orders(cumulative_bids)  
        filtered_asks = filter_orders(cumulative_asks)  

        return filtered_bids, filtered_asks  

    @staticmethod  
    def filter_abnormal_orders(  
        bids: List[Dict], asks: List[Dict], volume_threshold: float = 5  
    ) -> Tuple[List[Dict], List[Dict]]:  
        """  
        过滤异常巨量挂单。  
        :param bids: 买单列表，每个元素是 {"price": ..., "volume": ...}  
        :param asks: 卖单列表，每个元素是 {"price": ..., "volume": ...}  
        :param volume_threshold: 异常挂单量的倍数阈值（相对于平均值）  
        :return: 过滤后的买单和卖单列表  
        """  
        avg_bid_volume = sum([order["volume"] for order in bids]) / len(bids) if bids else 0  
        avg_ask_volume = sum([order["volume"] for order in asks]) / len(asks) if asks else 0  

        filtered_bids = [  
            order for order in bids if order["volume"] <= avg_bid_volume * volume_threshold  
        ]  
        filtered_asks = [  
            order for order in asks if order["volume"] <= avg_ask_volume * volume_threshold  
        ]  

        return filtered_bids, filtered_asks  

    @staticmethod  
    def filter_outlier_prices(  
        bids: List[Dict], asks: List[Dict], median_price: float, price_deviation_threshold: float = 0.1  
    ) -> Tuple[List[Dict], List[Dict]]:  
        """  
        过滤价格异常的挂单。  
        :param bids: 买单列表，每个元素是 {"price": ..., "volume": ...}  
        :param asks: 卖单列表，每个元素是 {"price": ..., "volume": ...}  
        :param median_price: 市场中位数价格  
        :param price_deviation_threshold: 价格偏离中位数的最大比例  
        :return: 过滤后的买单和卖单列表  
        """  
        if median_price == 0:  
            return bids, asks  # 避免除0情况  

        filtered_bids = [  
            order  
            for order in bids  
            if abs(order["price"] - median_price) / median_price <= price_deviation_threshold  
        ]  
        filtered_asks = [  
            order  
            for order in asks  
            if abs(order["price"] - median_price) / median_price <= price_deviation_threshold  
        ]  

        return filtered_bids, filtered_asks  