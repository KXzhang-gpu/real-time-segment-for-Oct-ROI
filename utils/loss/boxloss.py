import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.parallel
import torch.utils.data.distributed
import cv2


class Projection_Loss(nn.Module):
    def __init__(self, n_classes):
        super(Projection_Loss, self).__init__()
        self.n_classes = n_classes

    def _dice_coefficient(self, x, target):
        """
        Dice Loss: 1 - 2 * (intersection(A, B) / (A^2 + B^2))
        :param x:
        :param target:
        :return:
        """
        eps = 1e-5
        n_inst = x.size(0)
        x = x.reshape(n_inst, -1)
        target = target.reshape(n_inst, -1)
        intersection = (x * target).sum(dim=1)
        union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
        loss = 1. - (2 * intersection / union)
        return loss

    def _compute_project_term(self, mask_snodes, gt_bitmasks):
        """
           compute projection loss
           mask_snodes,gt_bitmasks(BW) tensor [b,c,h,w]
        """
        mask_losses_y = self._dice_coefficient(
            mask_snodes.max(dim=2, keepdim=True)[0],
            gt_bitmasks.max(dim=2, keepdim=True)[0]
        )
        mask_losses_x = self._dice_coefficient(
            mask_snodes.max(dim=3, keepdim=True)[0],
            gt_bitmasks.max(dim=3, keepdim=True)[0]
        )
        projection_loss = (mask_losses_x + mask_losses_y).mean()
        return projection_loss

    def forward(self, inputs, target):
        loss = 0.0
        for i in range(0, self.n_classes):
            projection_loss = self._compute_project_term(
                inputs[i, :, :, :].reshape(1, inputs.shape[1], inputs.shape[2], inputs.shape[3]),
                target[i, :, :, :].reshape(1, target.shape[1], target.shape[2], target.shape[3]))
            loss += projection_loss
        return loss / self.n_classes


class Pairwise_Loss(nn.Module):
    def __init__(self, n_classes, threshould=0.9):
        super(Pairwise_Loss, self).__init__()
        self.n_classes = n_classes
        self.threshold = threshould

    def graph_edge_generate(self, x, kernel_size, dilation):
        """
           according to figure2, generate the edge of the image graph
           node tensor [b,c,the number of node,1]
           edge tensor [b,c,the number of node,the number of neighbour]
        """
        padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
        unfolded_x = F.unfold(
            x, kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        unfolded_x = unfolded_x.reshape(x.size(0), x.size(1), unfolded_x.size(2), unfolded_x.size(1))
        size = kernel_size ** 2
        node_index = size // 2 + 1
        node = unfolded_x[:, :, :, int(node_index)]
        node = node.reshape(x.size(0), x.size(1), unfolded_x.size(2), 1)
        edge = torch.cat(
            (unfolded_x[:, :, :, :size // 2], unfolded_x[:, :, :, size // 2 + 1:]), dim=3)
        return edge, node

    def _get_image_color_similarity(self, image, pairwise_size, pairwise_dilation):
        Edge_image, node_image = self.graph_edge_generate(
            image, kernel_size=pairwise_size, dilation=pairwise_dilation
        )
        node_image = torch.repeat_interleave(node_image, Edge_image.size(3), dim=3)
        diff = node_image - Edge_image
        similarity = torch.exp(-torch.abs(diff) * 0.5)
        return similarity

    def _compute_pairwise_term(self, mask_scores, similarity, threshold, pairwise_size, pairwise_dilation):

        # we compute the probability in log space to avoid numerical instability
        # mask_scores = F.logsigmoid(mask_scores)

        sigmoid = torch.nn.Sigmoid()
        mask_scores = sigmoid(mask_scores)
        edge_mask, node_mask = self.graph_edge_generate(
            mask_scores, kernel_size=pairwise_size, dilation=pairwise_dilation
        )
        node_mask = torch.repeat_interleave(node_mask, edge_mask.size(3), dim=3)
        probability = torch.mul(edge_mask, node_mask) + torch.mul(1 - edge_mask, 1 - node_mask)
        zero = torch.zeros_like(similarity)
        one = torch.ones_like(similarity)
        flag = torch.where(similarity >= threshold, one, similarity)
        flag = torch.where(flag < threshold, zero, flag)
        P_with_weight = torch.mul(flag, torch.log(probability))
        P_with_weight = torch.squeeze(P_with_weight)
        P_with_weight = torch.sum(torch.sum(P_with_weight, dim=0), dim=0)
        pairwise_loss = -P_with_weight / (edge_mask.size(2))
        return pairwise_loss

    def forward(self, inputs, image, threshold=0, pairwise_size=3, pairwise_dilation=2):
        if threshold == 0:
            threshold = self.threshold
        loss = 0.0
        for i in range(0, self.n_classes):
            similarity = self._get_image_color_similarity(image[i, :, :, :].reshape(1, image.shape[1], image.shape[2],
                                                                                    image.shape[3]),
                                                          pairwise_size, pairwise_dilation)
            pairwise_loss = self._compute_pairwise_term(inputs[i, :, :, :].reshape(1, inputs.shape[1], inputs.shape[2],
                                                                                   inputs.shape[3])
                                                        , similarity, threshold, pairwise_size, pairwise_dilation)
            loss += pairwise_loss
        return loss / self.n_classes




if __name__ == '__main__':
    test_image = np.load(
        'C:/Users/27019/Desktop/shuyan/Annotation-efficient-learning-for-OCT-segmentation-main/Annotation-efficient'
        '-learning-for-OCT-segmentation-main/utils/new/image(1).npy', encoding="latin1")

    label = np.load(
        'C:/Users/27019/Desktop/shuyan/Annotation-efficient-learning-for-OCT-segmentation-main/Annotation-efficient'
        '-learning-for-OCT-segmentation-main/utils/new/label(1).npy', encoding="latin1")

    test_image = torch.Tensor(test_image)
    test_image = test_image.reshape(1, 1, 1500, 2250)
    label = torch.Tensor(label)
    label = label.reshape(1, 1, 1500, 2250)
    test_mask = torch.zeros(1, 1, 1500, 2250)
    test_mask[0, 0, 600:700, 1400:1600] = 1
    Projection_loss_test = Projection_Loss(1)
    loss1 = Projection_loss_test(test_mask, label)
    print(loss1)

    Pairwise_loss_test = Pairwise_Loss(1)
    loss2 = Pairwise_loss_test(label, test_image, 0.9)
    print(loss2)

    test_OCT = cv2.imread('C:/Users/27019/Desktop/shuyan/Annotation-efficient-learning-for-OCT-segmentation-main'
                          '/Annotation-efficient-learning-for-OCT-segmentation-main/utils/new/48.png', cv2.IMREAD_GRAYSCALE)
    label_OCT = cv2.imread('C:/Users/27019/Desktop/shuyan/Annotation-efficient-learning-for-OCT-segmentation-main'
                           '/Annotation-efficient-learning-for-OCT-segmentation-main/utils/new/48_gt.png', cv2.IMREAD_GRAYSCALE)

    test_OCT = torch.Tensor(test_OCT)
    test_OCT = test_OCT.reshape(1, 1, 512, 512)
    label = torch.Tensor(label_OCT)
    label = label.reshape(1, 1, 512, 512)

    
