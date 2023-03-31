import logging


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(filename)s - %(levelname)s: \n%(message)s \n\n',
        handlers=[
            logging.FileHandler('experiment.log'),
            logging.StreamHandler()
        ]
    )


"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s - %(filename)s - %(levelname)s \n Message :: %(message)s')

file_handler = logging.FileHandler('logs.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
"""
