import os

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from src.logger import setup_logger
    setup_logger('main.log')
    import manager
    manager.start()
