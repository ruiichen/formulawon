from consts import API_VERSION
from http import HTTPStatus
from requests import HTTPError, Response

def exception_response(error):
    return {"apiVersion": API_VERSION, "error": {"code": error.response.status_code, "message": error.message}}

class ExternalServiceError(HTTPError):
    def __init__(self):
        super()
        self.response = Response()
        self.response.status_code = HTTPStatus.SERVICE_UNAVAILABLE.value
        self.message = "ERROR: The Ergast API was unresponsive."

class NotFoundError(HTTPError):
    def __init__(self):
        super()
        self.response = Response()
        self.response.status_code = HTTPStatus.NOT_FOUND.value
        self.message = "ERROR: The race was not found."

class InternalServerError(HTTPError):
    def __init__(self):
        super()
        self.response = Response()
        self.response.status_code = HTTPStatus.INTERNAL_SERVER_ERROR.value
        self.message = "ERROR: Something went wrong 🎀."