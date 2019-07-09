from .registry import DATASETS
from .xml_style import XMLDataset

@DATASETS.register_module
class FaultDetection(XMLDataset):
	CLASSES = ('油污','鸟巢','锈蚀','飘挂物')

	def __init__(self, **kwargs):
		super(FaultDetection, self).__init__(**kwargs)