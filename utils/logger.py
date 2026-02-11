import logging
from typing import Optional

console_formatter_str = "[\033[032m%(asctime)s\033[0m %(levelname)s] %(pathname)s:%(lineno)d %(message)s"
file_formatter_str = "[%(asctime)s %(levelname)s] %(threadName)s %(pathname)s:%(lineno)d %(message)s"

def init_logger(
    log_file: Optional[str] = None,
    log_file_level=logging.NOTSET,
    log_level=logging.INFO,
):
    """
    Initializes and configures the root logger.

    Args:
        log_file (Optional[str]): The path to the log file. If None, no file handler is added.
        log_file_level (int): The logging level for the file handler.
        log_level (int): The logging level for the console handler.
    """
    # 1. Clear any existing handlers from the root logger
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 2. Set the overall log level
    root_logger.setLevel(logging.DEBUG)  # Set to DEBUG to ensure all messages are captured

    # 3. Create a formatter for both handlers
    console_formatter = logging.Formatter(console_formatter_str)
    file_formatter = logging.Formatter(file_formatter_str)

    # 4. Create and configure the console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 5. Create and configure the file handler if a log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # 6. Return the root logger for immediate use
    return root_logger

def get_root_log_file_path():
    root_logger_handlers = logging.getLogger().handlers
    for handler in root_logger_handlers:
        if "FileHandler" in handler.__class__.__name__:
            return handler.baseFilename
    raise ValueError("No file handler found")
    # return None

def get_logger(name, file_level=logging.INFO, console_level=logging.INFO):

    """
    Creates and returns a new logger with the specified name and level.

    Args:
        name (str): The name of the logger.
        level (int): The logging level for the logger.

    Returns:
        logging.Logger: The configured logger instance.
    """

    if name in logging.Logger.manager.loggerDict:

        logger = logging.getLogger(name)
        return logger

    logger = logging.getLogger(name)

    logger.setLevel(min(file_level, console_level))
    logger.propagate = False


    # add stream handler and file handler
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(level)
    # logger.addHandler(console_handler)

    if logger.hasHandlers():
        logger.handlers.clear()

    log_file_path = get_root_log_file_path()

    console_formatter = logging.Formatter(console_formatter_str)
    file_formatter = logging.Formatter(file_formatter_str)

    # 4. Create and configure the console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_file_path:

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # if logger.name == "utils.shift_simulate":
    #     print("The information about logger in utils.shift_simulate:")
    #     print(logger.level)
    #     print(logger.handlers)

    return logger
