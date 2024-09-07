import torch


def pred(w, b, x):
    return torch.matmul(w, x.T) + b


def loss(w, b, x, y):
    return (y - pred(w, b, x)).pow(2).mean()


# 가중치를 업데이트 하는 함수
def grad_w(w, b, x, y):
    # w: (1, d), b: (1, 1), x: (n, d), y: (n)
    tmp1 = torch.matmul(w, x.T)  # (1, n)
    tmp2 = tmp1 + b  # (1, n)
    # 미분하여 2(yw - yi) * x 가 나옴
    tmp3 = 2 * (tmp2 - y[None])  # (1, n)
    grad_item = tmp3.T * x  # (n, d)

    return grad_item.mean(dim=0, keepdim=True)  # (1, d)


# 편향을 업데이트 하는 함수
def grad_b(w, b, x, y):
    # w: (1, d), b: (1, 1), x: (n, d), y: (n)
    grad_item = 2 * (torch.matmul(w, x.T) + b - y[None])  # (1, n)
    return grad_item.mean(dim=-1, keepdim=True)  # (1, 1)


def update(x, y, w, b, lr):
    w = w - lr * grad_w(w, b, x, y)
    b = b - lr * grad_b(w, b, x, y)
    return w, b


# epochs 만큼 학습
# 단계당 가중치와 편향을 업데이트함
# 업데이트 된 값으로 오차를 계산하여 예측 값을 확인
def train(n_epochs, lr, w, b, x, y):
    for e in range(n_epochs):
        w, b = update(x, y, w, b, lr)
        print(f"Epoch {e:3d} | Loss: {loss(w, b, x, y)}")
    return w, b


x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([0, 1, 1, 1])

print(x.shape, y.shape)

w = torch.randn((1, 2))
b = torch.randn((1, 1))

print(w.shape, b.shape)

n_epochs = 10
lr = 0.1

w, b = train(n_epochs, lr, w, b, x, y)

print(pred(w, b, x))
print(y)
