### --- IMPORTS --- ###
import logging
from pathlib import Path

##########################
log_file = Path(__file__).parent / "data" / "tutor.log"
log_file.parent.mkdir(parents=True, exist_ok=True)


def log_system() -> None:
    """creates and setup one system logs, where the user sees in terminal logs the type
    INFO or above and the function register internaly only
    types WARNING or above in file .log"""

    ######## --- USER --- #########
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    terminal_controller = logging.StreamHandler()
    terminal_controller.setLevel(logging.INFO)
    terminal_formated = logging.Formatter("%(message)s")
    terminal_controller.setFormatter(terminal_formated)
    logger.addHandler(terminal_controller)

    ######## --- FILE .LOG --- #########
    file_controller = logging.FileHandler(log_file, encoding="utf-8")
    file_controller.setLevel(logging.WARNING)
    file_formated = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
    )
    file_controller.setFormatter(file_formated)
    logger.addHandler(file_controller)
