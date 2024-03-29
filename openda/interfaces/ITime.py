class ITime:
    """
    Interface of Time

    @author Nils van Velzen
    """

    def __init__(self):
        """
        Empty constructor
        """

    def get_start(self):
        """
        Returns the start of the time period.

        :return: start time.
        """
        raise NotImplementedError("Function not implemented.")

    def get_end(self):
        """
        Returns the start of the time period.

        :return: start time.
        """
        raise NotImplementedError("Function not implemented.")

    def get_is_span(self):
        """
        Check whether self is a time span or a time stamp.

        :return: True if self is a time span.
        """
        raise NotImplementedError("Function not implemented.")

    def after(self, other_time):
        """
        Check whether self starts after other_time ends.

        :param other_time: time object to be compared
        :return: True if self starts after other_time ends.
        """
        raise NotImplementedError("Function not implemented.")

    def get_step_mjd(self):
        """
        Get the time step interval in days (as Modified Julian Day).
        :return The time step interval. Throw an exception if is is not available.
        """
        raise NotImplementedError("Function not implemented.")

    def get_mjd(self):
        """
        Returns a time stamp in the middle of the time period.

        :return: center of time period.
        """
        raise NotImplementedError("Function not implemented.")

    def __str__(self):
        if self.get_is_span():
            # pylint: disable=consider-using-f-string
            return "[%s, %s]"%(self.get_start(),self.get_end())
        return str(self.get_start())
