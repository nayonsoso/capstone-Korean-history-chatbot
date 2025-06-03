class BadRequestException(Exception):
    def __init__(self, message: str):
        self.message = message

class InternalServerException(Exception):
    def __init__(self, message: str):
        self.message = message
