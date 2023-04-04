import logging

# For another color of logging output:
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(filename)s - %(levelname)s: \n%(message)s \n\n',
        handlers=[
            logging.FileHandler('experiment.log'),
            logging.StreamHandler()
        ]
    )
