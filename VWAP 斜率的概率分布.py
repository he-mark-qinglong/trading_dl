# 右侧：VWAP 斜率的概率分布（含动态密度中心线）
if len(vwap_slopes) > 1:
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vwap_slopes)
        slope_values = np.linspace(min(vwap_slopes), max(vwap_slopes), 100)
        slope_density = kde(slope_values)
        
        # 新增：计算密度峰值位置
        peak_position = slope_values[np.argmax(slope_density)]
        
        # 新增：维护峰值轨迹（保留最近50个观测值）
        if not hasattr(self, 'peak_history'):
            self.peak_history = []
        self.peak_history.append(peak_position)
        self.peak_history = self.peak_history[-50:]  # 滑动窗口
        
        # 绘制概率密度曲线
        vwap_slope_fig.add_trace(
            go.Scatter(
                x=slope_density,
                y=slope_values,
                mode="lines",
                line=dict(color="blue", width=2),
                name="瞬时密度分布"
            ),
            row=1, col=2
        )
        
        # 新增：动态密度中心线
        vwap_slope_fig.add_trace(
            go.Scatter(
                x=np.linspace(0, max(slope_density), len(self.peak_history)),
                y=self.peak_history,
                mode="lines+markers",
                line=dict(color="#FFA500", width=1.5, dash="dot"),
                marker=dict(size=6, symbol="diamond"),
                name="密度中心轨迹",
                hovertemplate="峰值: %{y:.4f}<extra></extra>"
            ),
            row=1, col=2
        )
        
        # 绘制直方图（使用透明条显示分布范围）
        vwap_slope_fig.add_trace(
            go.Bar(
                x=vwap_slopes,
                y=[0]*len(vwap_slopes),
                marker=dict(
                    color="rgba(100, 149, 237, 0.3)",  # 浅蓝色透明
                    line=dict(width=0)  # 去除边框线
                ),
                name="斜率分布直方图",
                hoverinfo="skip"
            ),
            row=1, col=2
        )
        
        # 新增：零轴参考线
        vwap_slope_fig.add_hline(
            y=0, 
            line=dict(color="gray", width=1, dash="dash"),
            row=1, col=2
        )
        
    except Exception as e:
        print("VWAP斜率可视化异常:", e)

# 增强布局配置
vwap_slope_fig.update_layout(
    title="VWAP动态分析 (带密度中心轨迹)",
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    xaxis2=dict(title="概率密度", showgrid=False),
    yaxis2=dict(
        title="斜率值",
        range=[min(vwap_slopes)*1.1 if len(vwap_slopes)>0 else -0.1, 
               max(vwap_slopes)*1.1 if len(vwap_slopes)>0 else 0.1],
        zerolinecolor="gray"
    ),
    hovermode="x unified"
)