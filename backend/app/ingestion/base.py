from abc import ABC,abstractmethod
from pathlib import Path

from app.models.document import  NormalizedDocument

class DocumentLoader(ABC):
    @abstractmethod
    def load(
        self,
        file_path:str|Path,
        uploaded_by:str|None=None,
    )->NormalizedDocument:
        raise NotImplementedError