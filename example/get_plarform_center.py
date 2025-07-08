import numpy as np


# Platform 1
corner_1= np.array([1615.78, 1015.5, 9.72141])
corner_2= np.array([1621.78, 524.955, 10.4409])
corner_3= np.array([-151.846, 522.568, 0.154146])
corner_4= np.array([-147.782, 1013.3, 0.582982])

platform_center = (corner_1 + corner_2 + corner_3 + corner_4)/4
# out -> np.array([734.483    , 769.08075  ,   0])


# Platform 2
corner_1= np.array([1622.09, 501.015, 9.21894])
corner_2= np.array([1618.21, 7.29261, 6.36424])
corner_3= np.array([-152.233, 8.11329, 0.245193])
corner_4= np.array([-152.447, 500.369, 0.527547])

platform_center = (corner_1 + corner_2 + corner_3 + corner_4)/4
# out -> np.array([733.905   , 254.197475,   0])
