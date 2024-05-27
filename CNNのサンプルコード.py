import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# データの準備
transform = transforms.Compose([
    transforms.ToTensor(),  # 画像をTensorに変換
    transforms.Normalize((0.5,), (0.5,))  # ピクセル値を[-1, 1]の範囲に正規化
])

# 訓練データセットのロードとデータローダの作成
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# テストデータセットのロードとデータローダの作成
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# CNNの定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 畳み込み層1、入力チャネル1、出力チャネル32、カーネルサイズ3x3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 畳み込み層2、入力チャネル32、出力チャネル64、カーネルサイズ3x3
        self.conv2_drop = nn.Dropout2d(p=0.25)  # ドロップアウト層、ドロップアウト率25%
        self.fc1 = nn.Linear(9216, 128)  # 全結合層1、入力ユニット9216、出力ユニット128
        self.fc2 = nn.Linear(128, 10)  # 全結合層2、入力ユニット128、出力ユニット10（クラス数）

    def forward(self, x):
        x = self.conv1(x)  # 畳み込み層1を適用
        x = nn.ReLU()(x)  # 活性化関数ReLUを適用
        x = self.conv2(x)  # 畳み込み層2を適用
        x = nn.ReLU()(x)  # 活性化関数ReLUを適用
        x = nn.MaxPool2d(2)(x)  # 2x2の最大プーリングを適用
        x = self.conv2_drop(x)  # ドロップアウトを適用
        x = torch.flatten(x, 1)  # フラット化（1次元に変換）
        x = self.fc1(x)  # 全結合層1を適用
        x = nn.ReLU()(x)  # 活性化関数ReLUを適用
        x = nn.Dropout(p=0.5)(x)  # ドロップアウトを適用
        x = self.fc2(x)  # 全結合層2を適用
        return nn.LogSoftmax(dim=1)(x)  # 出力に対してLogSoftmaxを適用

# モデルのインスタンス化とオプティマイザ、損失関数の定義
model = Net()  # モデルのインスタンス化
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adamオプティマイザの設定、学習率0.001
criterion = nn.CrossEntropyLoss()  # 損失関数としてクロスエントロピー誤差を使用

# トレーニングループ
for epoch in range(10):  # エポック数を10に設定
    model.train()  # モデルを訓練モードに設定
    running_loss = 0.0  # 累積損失の初期化
    for i, data in enumerate(trainloader, 0):  # 訓練データローダからバッチごとにデータを取得
        inputs, labels = data  # 入力データとラベルに分割
        optimizer.zero_grad()  # 勾配の初期化
        outputs = model(inputs)  # モデルの順伝播
        loss = criterion(outputs, labels)  # 損失の計算
        loss.backward()  # 逆伝播による勾配の計算
        optimizer.step()  # オプティマイザによるパラメータの更新
        running_loss += loss.item()  # 累積損失の更新
        if i % 100 == 99:  # 100ミニバッチごとに進捗を表示
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0  # 累積損失のリセット

print('Finished Training')  # トレーニングの終了を表示

# テストループ
model.eval()  # モデルを評価モードに設定
correct = 0  # 正解数の初期化
total = 0  # 合計サンプル数の初期化
with torch.no_grad():  # 勾配計算を無効に
    for data in testloader:  # テストデータローダからデータを取得
        images, labels = data  # 画像とラベルに分割
        outputs = model(images)  # モデルの順伝播
        _, predicted = torch.max(outputs.data, 1)  # 最も確率の高いクラスを取得
        total += labels.size(0)  # 合計サンプル数を更新
        correct += (predicted == labels).sum().item()  # 正解数を更新

# テストデータに対する精度を表示
print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
