from .registry import DATASETS
from .xml_style import XMLDataset

@DATASETS.register_module
class DigitDetection(XMLDataset):
	CLASSES = ('0','1','2','3','4','5','6','7','8','9')
	def __init__(self, **kwargs):
		super(DigitDetection, self).__init__(**kwargs)