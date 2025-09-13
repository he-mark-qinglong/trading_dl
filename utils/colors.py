def initialize_colors():  
    return {  
        'volume': '#586994',  
        'vwap': '#f39c12',  
        'obv': '#2ecc71',  
        'obv_ma': '#e74c3c',  
        'cmf': '#8e44ad',  
        'force_index': '#3498db',  
        'pvt': '#e67e22',  
        'rsi': '#9b59b6',  
        'ma': ['#FF9800', '#2196F3', '#4CAF50', '#E91E63', '#9C27B0', '#795548', '#607D8B', '#F44336'],  
        'ema': ['#FF5722', '#1976D2', '#388E3C', '#C2185B', '#7B1FA2', '#5D4037', '#455A64', '#D32F2F'],  
        'boll': {'upper': '#00BCD4', 'middle': '#006064', 'lower': '#00BCD4', 'fill': '#E0F7FA'},  
        'macd': {'macd': '#2196F3', 'signal': '#FF5722', 'histogram': '#78909C'},  
        'pivot': {  
            'pivot': '#FFB74D', 'r1': '#FF7043', 'r2': '#F4511E', 'r3': '#D84315',  
            's1': '#66BB6A', 's2': '#43A047', 's3': '#2E7D32'  
        },  
        'fractal': {'high': '#FF5252', 'low': '#69F0AE'},  
        'zigzag': '#BA68C8',  
        'micro_structure': {  
            'spread': '#FF4081', 'imbalance': '#7C4DFF',  
            'depth': '#00BFA5', 'trade_size': '#FFC400'  
        },  
        'order_book': {  
            'bid': '#4CAF50', 'ask': '#F44336', 'imbalance': '#9C27B0',  
            'pressure': '#FF9800', 'liquidity': '#03A9F4'  
        }  
    }  