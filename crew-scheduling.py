import pandas as pd
import numpy as np
from rectpack import newPacker
import rectpack.packer as packer
import matplotlib.pyplot as plt

# Initialize Model Parameters

#-- Pallet Dimensions: 80 x 120 cm
bx = 5 # Buffer x
by = 5 # Buffer y
pal_812 = [80 + bx, 120 + by]
#-- Pallet Dimensions: 100 x 120 cm
bx = 5 # Buffer x
by = 5 # Buffer y
pal_1012 = [100 + bx, 120 + by]

# Container size
bins20 = [(235, 590)] # 20' Container
bins40 = [(235, 1203)] # 40' Container