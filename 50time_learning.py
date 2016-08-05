#!/usr/bin/env python
# coding:utf-8
import numpy as np
import chainer.functions as F
from chainer import Variable, FunctionSet, optimizers

#モデル定義
model = F.Linear(3,3)
optimizer = optimizers.SGD()
optimizer.setup(model)


#学習させる回数
times = 50


#与えるベクトル  これが[2,4,6]になって返ってきて欲しい
x = Variable(np.array([[1, 2, 3]], dtype=np.float32))

#正解ベクトルの[2,4,6]
t = Variable(np.array([[2, 4, 6]], dtype=np.float32))

#ここから50回ループ
for i in range(0,times):
    optimizer.zero_grads()

    #ここでモデルに予測させている
    y = model(x)

    #モデルが出した答えを表示
    print(y.data)

    #お馬鹿なモデルが出した答えと、本当の答え([2,4,6])がどのくらい違っているか計算する
    loss = F.mean_squared_error(y, t)

    #その値をモデルに見せて「全然違うじゃねーか！もっと近づけろ！」と学習させる
    loss.backward()
    optimizer.update()
    
    #最初に戻って繰り返す

