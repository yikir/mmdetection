from .voc import VOCDataset
class FaultDetection(VOCDataset):
	CLASSES = ('油污','鸟巢','锈蚀','飘挂物')