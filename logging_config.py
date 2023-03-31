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
