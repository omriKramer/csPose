from fastai.vision import *

import models.cs_v2 as cs
import pose
import utils
from utils import DataTime


class SelfObserveInstructor(cs.RecurrentInstructor):
    def __init__(self):
        super().__init__(2)

    def on_bu_pred_begin(self, model):
        if self.i == 0:
            return False

        return True

    def on_td_begin(self, model, img_features, last_bu, bu_out, td_out):
        if self.i == 0:
            return last_bu.new_ones((last_bu.shape[0], 16))
        preds = bu_out[-1].reshape(-1, 16, 3).argmax(dim=-1)
        wrong_preds = (preds == 1).float()
        return wrong_preds


class ErrorDetectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.heatmaps_bn = nn.BatchNorm2d(16)
        self.resnet = nn.Sequential(*list(models.resnet18(pretrained=False, num_classes=16 * 3)))
        self.resnet = nn.Sequential(*self.resnet[4:])
        self.resnet[0][0].conv1 = nn.Conv2d(16 + 64, 64, kernel_size=3, padding=1)

    def forward(self, img_features, td_out):
        out = self.heatmaps_bn(td_out)
        out = torch.cat((img_features, out), dim=1)
        out = self.resnet(out)
        return out


class InstructorObserver(cs.RecurrentInstructor):
    def __init__(self):
        super().__init__(2)
        self.error_net_out = None

    def on_init_end(self, model):
        model.error_detection_network = ErrorDetectionNet()

    def on_td_begin(self, model, img_features, last_bu, bu_out, td_out):
        if self.i == 0:
            return last_bu.new_ones((last_bu.shape[0], 16))

        error_pred = model.error_detection_network(img_features, td_out[-1])
        self.error_net_out = error_pred
        error_pred = error_pred.reshape(-1, 16, 3).argmax(dim=-1)
        error_pred = (error_pred == 1).float()
        return error_pred

    def on_forward_end(self, bu_out, td_out):
        error_net_out = self.error_net_out
        self.error_net_out = None
        return error_net_out, td_out


# mean head size of LIP validation set
default_threshold = 0.3314


class SelfCorrect:
    def __init__(self):
        self.detect_target = None
        self.is_wrong = None

    def correction_loss_func(self, error_out, heatmaps, targets):
        n = targets.shape[0]
        preds = pose.output_to_scaled_pred(heatmaps)
        first_td_preds = preds[:, :16]
        is_visible = targets[..., 2] > 0
        gt = targets[..., :2]

        head_sizes = torch.norm(gt[:, 8] - gt[:, 9], dim=1)
        thresholds = head_sizes / 2
        has_head = (is_visible[:, 8:10]).all(1)
        thresholds[~has_head] = default_threshold
        distances = torch.norm(first_td_preds - gt, dim=2)
        under_threshold = (distances < thresholds[:, None])
        is_correct = under_threshold * is_visible
        self.is_wrong = (~under_threshold) * is_visible

        detect_target = torch.zeros(n, 16, dtype=torch.long)
        detect_target[self.is_wrong] = 1
        detect_target[is_correct] = 2
        self.detect_target = detect_target.reshape(-1).to(targets.device)

        error_detect_loss = F.cross_entropy(error_out.reshape(-1, 3), self.detect_target)

        first_targets = gt[is_visible]
        first_td = heatmaps[:, :16][is_visible]
        pred_detect = error_out.reshape(-1, 16, 3).argmax(dim=2)
        pred_wrong = pred_detect == 1
        wrong = pred_wrong * is_visible
        second_targets = gt[wrong]
        second_td = heatmaps[:, 16:][wrong]
        td = torch.cat((first_td, second_td))
        td_targets = torch.cat((first_targets, second_targets))
        keypoints_loss = pose.ce_loss(td, td_targets)
        return error_detect_loss + keypoints_loss

    def loss_func(self, outputs, targets):
        error_out, td_out = outputs
        return self.correction_loss_func(error_out, td_out, targets)

    def accuracy(self, outputs, targets):
        bu_out = outputs[1][-1].reshape(-1, 3)
        return accuracy(bu_out, self.detect_target)

    def heatmap_func(self, outputs):
        heatmaps = outputs[1]
        n, _, h, w = heatmaps.shape
        combined = torch.empty(n, 16, h, w).to(heatmaps.device)
        combined[self.is_wrong] = heatmaps[:, 16:][self.is_wrong]
        combined[~self.is_wrong] = heatmaps[:, :16][~self.is_wrong]
        return torch.cat((heatmaps, combined), dim=1)


def main(args):
    print(args)
    arch = pose.nets[args.resnet]

    instructor = SelfObserveInstructor()

    root = Path(__file__).resolve().parent.parent / 'LIP'
    db = pose.get_data(root, args.size, bs=args.bs)

    self_correct = SelfCorrect()
    pckh = partial(pose.Pckh, niter=3, mean=False, heatmap_func=self_correct.heatmap_func)
    learn = cs.cs_learner(db, arch, instructor, td_c=16, bu_c=16 * 3, pretrained=False, embedding=None,
                          add_td_out=True, loss_func=self_correct.loss_func, metrics=self_correct.accuracy,
                          callback_fns=[pckh, DataTime])

    monitor = 'Total_1'
    utils.fit_and_log(learn, args, monitor)


if __name__ == '__main__':
    parser = utils.basic_train_parser()
    main(parser.parse_args())
