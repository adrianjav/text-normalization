import torch


class Accuracy(torch.nn.Module):
    def __init__(self, reduce=True):
        super(Accuracy, self).__init__()
        self.reduce = reduce

    def forward(self, input, target):
        indexes = torch.max(input, dim=2)[1]
        r = (indexes == target).sum(1).float()

        return r.mean() if self.reduce else r


class CERLoss(torch.nn.Module):
    def __init__(self, ignore_index=None, reduce=True):
        super(CERLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, input, target):
        indexes = torch.max(input, dim=2)[1]
        weights = (target != self.ignore_index).float()
        memo = input.new_ones((input.size()[0], input.size()[1]+1, 2))

        memo[:,0,0] = 0
        memo[:,:,0] = memo[:,:,0].cumsum(1)
        for i in range(1, target.size()[1]+1):
            memo[:, 0, i % 2] = memo[:, 0, (i+1) % 2] + 1 * weights[:, i-1]
            for j in range(1, input.size()[1]+1):
                memo[:, j, i % 2] = (1 - weights[:, i-1]) * memo[:, j, (i+1) % 2] + weights[:, i-1] * torch.min(
                    memo[:, j, (i+1) % 2] + 1,
                    torch.min(
                        memo[:, j - 1, i % 2] + 1,
                        memo[:, j - 1, (i+1) % 2 ] + (target[:, i-1] != indexes[:, j-1]).float(),
                    )
                )

        loss = memo[:,:,target.size()[1] % 2].gather(1,weights.sum(1).long().unsqueeze(1)).squeeze(1)
        return loss.mean() if self.reduce else loss
