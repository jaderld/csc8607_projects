import torch
import torch.nn as nn

# ----------------------------
# Bottleneck Block
# ----------------------------
class BottleneckBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + identity
        out = self.relu(out)
        return out

# ----------------------------
# BottleneckNet
# ----------------------------
class BottleneckNet(nn.Module):
    def __init__(self, B=(2,2,2), bottleneck_mid=16, num_classes=100):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_stage(64, 64, bottleneck_mid, B[0], first_stride=1)
        self.layer2 = self._make_stage(64, 128, bottleneck_mid*2, B[1], first_stride=2)
        self.layer3 = self._make_stage(128, 256, bottleneck_mid*4, B[2], first_stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, num_classes)

    def _make_stage(self, in_ch, out_ch, mid_ch, blocks, first_stride):
        layers = [BottleneckBlock(in_ch, mid_ch, out_ch, stride=first_stride)]
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(out_ch, mid_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ----------------------------
# build_model
# ----------------------------
def build_model(config: dict):
    B = config.get("blocks", [2,2,2])
    bottleneck_mid = config.get("bottleneck_mid", 16)
    num_classes = config.get("num_classes", 100)
    return BottleneckNet(B=B, bottleneck_mid=bottleneck_mid, num_classes=num_classes)