import torch

class SampleNet(torch.nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()

    def forward(self, batch_idx, topk_idx):
        batch_idx = batch_idx.byte()
        topk_idx = topk_idx.long()
        box_regression = torch.randn(1, 40800, 4)
        print(batch_idx)
        print(topk_idx.shape)
        print(box_regression.shape)
        box_regression = torch.index_select(box_regression, 1, topk_idx[0])
        print(box_regression.shape)
        return torch.index_select(box_regression, 1, topk_idx[0])


if __name__ == "__main__":
    net = SampleNet()
    batch_idx = torch.zeros(1, 1)
    topk_idx = torch.zeros(1, 1000)
    torch.onnx.export(net, (batch_idx, topk_idx), "sample.onnx", verbose=True)

#     ensor([[0]])
# torch.Size([1, 1000])
# torch.Size([1, 40800, 4])