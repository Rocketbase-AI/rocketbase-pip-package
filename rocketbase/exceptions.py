class RocketNotEnoughInfo(Exception):
    """
    Exception raised when not enough information is provided to fetch/find a Rocket.
    """

    def __init__(self, message, errors=None):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class RocketInfoFormat(Exception):
    """
    Exception raised when the format of a Rocket's information is not conform.
    """

    def __init__(self, message, errors=None):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class RocketAPIError(Exception):
    """
    Exception raised when the RocketAPI is returning an error code.
    """

    def __init__(self, message, errors=None):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class RocketNotFound(Exception):
    """
    Exception raised when no Rocket is found.
    """

    def __init__(self, message, errors=None):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class RocketHashNotValid(Exception):
    """
    Exception raised when the hash of a Rocket is not valid.
    """

    def __init__(self, message, errors=None):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

class RocketDeviceNotFound(Exception):
    """
    Exception raised when the device requested to run the Rocket doesn't exist.
    """

    def __init__(self, message, errors=None):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

class CloudStorageCredentials(Exception):
    """
    Exception raised when the retrieval of the credentials for the Cloud Storage failed
    """

    def __init__(self, message, errors=None):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

class ShadowRocketPostprocessData(Exception):
    """
    Exception raised when there are some missing information to successfully complete the post-process function of a ShadowRocket.
    """

    def __init__(self, message, errors=None):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)
