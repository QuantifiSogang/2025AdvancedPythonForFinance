
__docformat__ = 'restructuredtext en'
__author__ = "<Tommy Lee>"
__all__ = ['DataLoadError']

class DataLoadError(Exception):  # error 예외 처리를 위한 class 정의
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)