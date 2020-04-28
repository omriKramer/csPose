import fastai.vision as fv
import matplotlib.pyplot as plt


class ObjectAndParts(fv.ItemBase):

    def __init__(self, objects: fv.ImageSegment, parts: fv.ImageSegment):
        assert objects.shape == parts.shape
        self.objects = objects
        self.parts = parts

    @property
    def data(self):
        return self.objects.data, self.parts.data

    def apply_tfms(self, tfms, **kwargs):
        objects = self.objects.apply_tfms(tfms, **kwargs)
        parts = self.parts.apply_tfms(tfms, **kwargs)
        return self.__class__(objects, parts)

    def __repr__(self):
        return f'{self.__class__.__name__} {tuple(self.objects.size)}'


class ObjectsPartsLabelList(fv.ItemList):
    _processor = fv.data.SegmentationProcessor

    def __init__(self, items, classes=None, **kwargs):
        super().__init__(items, **kwargs)
        self.copy_new.append('classes')
        self.classes = classes

    def get(self, i):
        object_fn, parts_fn, adapter = super().get(i)
        obj, parts = adapter.open(object_fn, parts_fn)
        obj = fv.ImageSegment(obj)
        parts = fv.ImageSegment(parts)
        return ObjectAndParts(obj, parts)

    def analyze_pred(self, pred):
        raise NotADirectoryError

    def reconstruct(self, t, x=None):
        obj = fv.ImageSegment(t[0])
        parts = fv.ImageSegment(t[1])
        return ObjectAndParts(obj, parts)


class ObjectsPartsItemList(fv.ImageList):
    _label_cls = ObjectsPartsLabelList

    def show_xys(self, xs, ys, imgsize=4, figsize=None, **kwargs):
        rows = len(xs)
        axs = fv.subplots(rows, 2, imgsize=imgsize, figsize=figsize)
        for x, y, ax_row in zip(xs, ys, axs):
            x.show(ax=ax_row[0], y=y.objects, **kwargs)
            x.show(ax=ax_row[1], y=y.parts, **kwargs)
        plt.tight_layout()
