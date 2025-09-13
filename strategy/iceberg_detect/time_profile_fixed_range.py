//@version=5
indicator("Time Profile Fixed Range", overlay=true, max_boxes_count=1500)

// —— 参数区 —— 
length        = input.int(500, "Range Length (bars)",    minval=10,  maxval=1500)
buckets       = input.int(100,  "Number of Buckets",      minval=10,  maxval=200)
maxWidth      = input.int(50,  "Max Bar Width (px)",     minval=10,  maxval=200)
barsFromNow   = input.int(1,   "Endpoint Offset (bars)", minval=0,   maxval=500)
rangeBorderC  = input.color(color.gray,           "Border Color")
fillColor     = input.color(color.new(color.blue, 60), "Fill Color")

// —— 计算区间高低 —— 
anchorBar = bar_index - barsFromNow
startBar  = math.max(anchorBar - length + 1, 0)
highP     = ta.highest(high, length)[barsFromNow]
lowP      = ta.lowest( low, length)[barsFromNow]
hgt       = highP - lowP
bucketH   = hgt / buckets

// —— 全局数组/盒子/线 —— 
var float[] timeArr  = array.new_float(buckets, 0.0)
var float[] lowArr   = array.new_float(buckets, 0.0)
var float[] highArr  = array.new_float(buckets, 0.0)
var box[]   boxArr   = array.new_box( buckets, na )
var line    borderL  = na
var line    highLine = na
var line    lowLine  = na

// 每次重画前删掉旧盒子
if barstate.islast
    for b = 0 to array.size(boxArr) - 1
        box.delete(array.get(boxArr, b))

if barstate.islast
    // 1) 初始化桶边界 & 时间累加清零
    for i = 0 to buckets - 1
        array.set(lowArr,  i, lowP  + bucketH * i)
        array.set(highArr, i, lowP  + bucketH * (i + 1))
        array.set(timeArr,  i, 0.0)

    // 2) 扫描每根 K 线，把「1 单位时间」按价格重叠度分桶
    for k = 0 to length - 1
        idx = barsFromNow + (length - 1 - k)
        if idx >= 0 and idx <= bar_index
            bh = high[idx] - low[idx]
            if bh > 0
                for j = 0 to buckets - 1
                    lo = array.get(lowArr,  j)
                    hi = array.get(highArr, j)
                    ov = math.max(0.0, math.min(hi, high[idx]) - math.max(lo, low[idx]))
                    if ov > 0
                        array.set(timeArr, j, array.get(timeArr, j) + ov / bh)

    // 3) 找最大值归一化宽度
    float maxT = 0.0
    for j = 0 to buckets - 1
        maxT := math.max(maxT, array.get(timeArr, j))

    // 4) 画剖面边框线
    if not na(borderL)
        line.delete(borderL)
    borderL := line.new(startBar, lowP, anchorBar, lowP, xloc=xloc.bar_index, color=rangeBorderC)
    line.set_xy1(borderL, startBar, lowP)
    line.set_xy2(borderL, anchorBar, lowP)
    line.new(startBar, highP, anchorBar, highP, xloc=xloc.bar_index, color=rangeBorderC)

    // 5) 画最高/最低价水平线
    if na(highLine)
        highLine := line.new(startBar, highP, anchorBar, highP, xloc=xloc.bar_index, color=color.yellow, width=2)
    else
        line.set_xy1(highLine, startBar, highP)
        line.set_xy2(highLine, anchorBar, highP)

    if na(lowLine)
        lowLine := line.new(startBar, lowP, anchorBar, lowP, xloc=xloc.bar_index, color=color.blue, width=2)
    else
        line.set_xy1(lowLine, startBar, lowP)
        line.set_xy2(lowLine, anchorBar, lowP)

    // 6) 绘制各桶时间直方
    for j = 0 to buckets - 1
        lo = array.get(lowArr,  j)
        hi = array.get(highArr, j)
        t  = array.get(timeArr, j)
        w  = maxT > 0 ? int((t / maxT) * maxWidth) : 0  
        array.set(boxArr, j, box.new(left=anchorBar, right=anchorBar + int(w), top=hi, bottom=lo, xloc=xloc.bar_index, bgcolor=fillColor, border_color=rangeBorderC))