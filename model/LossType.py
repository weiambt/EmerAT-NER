from enum import Enum

# 损失类型
class LossType(Enum):
    AT = "0"
    FOCAL = "1"
    BALANCE_advloss = "2"
    BALANCE_crfloss = "3"
