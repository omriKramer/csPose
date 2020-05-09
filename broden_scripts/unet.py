from fastai.vision import *
import parts
import utils

from fastai.callbacks import SaveModelCallback, CSVLogger

broden_root = Path('unifiedparsing/broden_dataset').resolve()
data = parts.get_data(broden_root)

tree = parts.ObjectTree.from_meta_folder(broden_root / 'meta')
n_obj = tree.n_obj


def split_pred(last_output):
    return last_output[:, :n_obj], last_output[:, n_obj:]


c = tree.n_obj + tree.n_parts

loss = parts.Loss(tree, preds_func=split_pred)
metrics = partial(parts.BrodenMetrics, obj_tree=tree, preds_func=split_pred)

learn = unet_learner(data,
                     models.resnet50,
                     loss_func=loss,
                     callback_fns=[metrics, utils.DataTime])

lr = 2e-4
learn.fit_one_cycle(5, lr, callbacks=[
    SaveModelCallback(learn, monitor='object-P.A.', name='unet-stage1'),
    CSVLogger(learn, filename='unet-stage1')]
                    )
learn.unfreeze()
learn.fit_one_cycle(10, slice(1e-6, lr / 5), callbacks=[
    SaveModelCallback(learn, monitor='object-P.A.', name='unet-stage2'),
    CSVLogger(learn, filename='unet-stage2')]
                    )
