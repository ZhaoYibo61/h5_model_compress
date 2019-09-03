import numpy as np
import pandas as pd
####生成9000,0000条数据，9千万条
a = np.random.standard_normal((100000,4))
b = pd.DataFrame(a)
####普通格式存储：
h5 = pd.HDFStore('/home/ding/test_s.h5','w')
h5['data'] = b
h5.close()