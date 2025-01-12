
def print_vars(logger,variable):
    logger.info("==========================================")
    # 获取所有变量名（属性名），过滤掉内置属性和方法
    defined_vars = [var for var in dir(variable) if not var.startswith('__') and not callable(getattr(variable, var))]

    # 遍历变量名，打印变量名和对应的值
    for var in defined_vars:
        value = getattr(variable, var)
        logger.info(f"{var}: {value}")
    logger.info("==========================================")