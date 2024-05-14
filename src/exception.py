import sys
from src.logger import logging

def error_message_details(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message =f"Error Occured In script {file_name}"

    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super.__init__(error_message)
        self.error_message = error_message_details(error=error_message,error_detail=error_details)

    def __str__(self):
        return self.error_message


if __name__ == '__main__':
    try:
        x=1/0
    except Exception as E:
        logging.info('Divide by 0 exception')