import logging

def get_logger(process_floder_path, name):
    logger = logging.getLogger(name)
    filename = f'{process_floder_path}/{name}.log'

    # 追加模式：恢复训练时不会清空已有日志，新内容接在旧内容后面。
    # 若同一进程内重复调用（同名 logger），直接复用已有的 handler，避免重复写行。
    if logger.handlers:
        return logger

    fh = logging.FileHandler(filename, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
