import abc

class InvertedIndex(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'search_doc') and
                callable(subclass.search_doc) and
                hasattr(subclass, 'index_doc') and
                callable(subclass.index_doc) and
                hasattr(subclass, 'update_doc') and
                callable(subclass.update_doc))
