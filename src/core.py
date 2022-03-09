import configparser

class IModule():
    """
    Module interface describing the 
    """
    def __init__(self):
        super().__init__()
        if self.__class__.__name__ == "IModule":
            raise Exception("This is an interface that cannot be instantiated")
            
    def transform(self, inputs):
        raise NotImplementedError(f"Method transform of {self.__class__.__name__} was not implemented")


class ConfigParserDict(configparser.ConfigParser):
    """
    ini to dict from: https://stackoverflow.com/questions/3220670/read-all-the-contents-in-ini-file-into-dictionary-with-python
    
    changed by: Tiago Almeida
    """
    def read_file(self, f, source=None):
        """
        overload of the read_file method to convert the ini to python dict
        """
        super().read_file(f, source)
        d = dict(self._sections)
        for k in d:
            d[k] = dict(self._defaults, **d[k])
            d[k].pop('__name__', None)
        return d
