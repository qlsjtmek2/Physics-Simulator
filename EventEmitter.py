from abc import ABC, abstractmethod


class EventEmitter(ABC):
    """이벤트가 발생할 수 있는 객체"""
    @abstractmethod
    def add_event_listener(self, event_name, event_listener):
        """
        이벤트 리스너를 추가합니다.

        Parameters:
            event_name (str): 이벤트의 이름
            event_listener (callable): 이벤트를 처리하는 함수 또는 메서드
        """

    @abstractmethod
    def notify_event_listeners(self, event_name, event_object):
        """
        등록된 이벤트 리스너에게 이벤트를 알립니다.

        Parameters:
            event_name (str): 이벤트의 이름
            event_object (event Object): 이벤트에 대한 정보가 담긴 이벤트 객체
        """
