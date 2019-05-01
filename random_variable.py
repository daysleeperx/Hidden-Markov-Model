"""Random Variable."""


class RandomVariable:
    """
    Variables in probability theory are called random variables and their names begin with an uppercase letter.
    Every random variable has a domain - the set of possible values it can take on.
    """
    def __init__(self, name: str, domain: list):
        """
        Class constructor.

        :param name:
        :param domain:
        """
        self._name = name
        self._domain = domain

    @property
    def name(self):
        """Name used to uniquely identify this variable."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def domain(self):
        """Set of possible values the Random Variable can take on."""
        return self._domain

    @domain.setter
    def domain(self, value):
        self._domain = value
