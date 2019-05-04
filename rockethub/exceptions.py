class RocketNotEnoughInfo(Exception):
    """ 
    Exception raised when not enough information is provided to fetch/find a Rocket.
    """
    def __init__(self, message, errors = None):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        pass

class RocketAPIError(Exception):
    """ 
    Exception raised when the RocketAPI is returning an error code.
    """
    def __init__(self, message, errors = None):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        pass

class RocketNotFound(Exception):
    """ 
    Exception raised when no Rocket is found.
    """
    def __init__(self, message, errors = None):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        pass