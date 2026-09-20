#- Neuron, layer, parameter ve loss kavramları
#- Forward pass mantığı
#- Modelin öğrenmesi ne demektir
#- Gradient descent sezgisi

import matplotlib.pyplot as plt


# yalniz bir nöron
def neuron(x1, x2, w1, w2, b):
    z = w1 * x1 +   w2 * x2 + b
    return max(0, z)    # ReLU


# tek nöron ile forward pass 
single_output = neuron(2, 3, 1, 1, 0)


print("tek bir nöronun forward pass ciktisi ", single_output)


def layer(x1, x2, weights, biases):
    outputs = []

    for w, b in zip(weights, biases):
        y = neuron(x1, x2, w[0], w[1], b)
        outputs.append(y)

    return outputs


# Katman testi
weights = [
    [1, 1],
    [0.5, 1],
    [-1, 0.5]
]

biases = [0, 1, 0]

layer_output = layer(2, 3, weights, biases)


print("Katmandaki noronlar:", layer_output)


# Loss
def loss(prediction, target):
    return (prediction - target) ** 2


# Örnek veri
x1 = 2
x2 = 3
target = 10

# Başlangıç parametreleri
w1 = 1
w2 = 1
b = 0


# Loss testi
prediction = neuron(x1, x2, w1, w2, b)
current_loss = loss(prediction, target)


print("Prediction:", prediction)
print("Target:", target)
print("Loss:", current_loss)



# farlki loss degerlerini gozlemleme
values_of_w = []
losses = []

for w in  [i / 10 for i in range(0,  101)]:

    prediction = neuron(x1, x2, w, w2, b)

    current_loss = loss(prediction, target)


    values_of_w.append(w)
    losses.append(current_loss)



print("w1 = 0 için loss:", losses[0])
print("w1 = 10 için loss:", losses[-1])

plt.plot(values_of_w, losses)
plt.xlabel("w1")
plt.ylabel("Loss")
plt.show()




learning_rate = 0.001
h = 0.0001

#gradient 

for i in range(100):

    prediction = neuron(x1, x2, w1, w2, b)
    current_loss = loss(prediction, target)

    # w1'in türevini yaklaşık olarak hesaplama
    loss_plus = loss(
        neuron(x1, x2, w1 + h, w2, b),
        target
    )

    loss_minus = loss(
        neuron(x1, x2, w1 - h, w2, b),
        target
    )

    gradient_w1 = (loss_plus - loss_minus) / (2 * h)

    # w1'i güncelle
    w1 = w1 - learning_rate * gradient_w1
    new_prediction = neuron(x1, x2, w1, w2, b)
    new_loss = loss(new_prediction, target)

    if i % 10 == 0:
        print(
            "step:", i,
            "w1:", round(w1, 4),
            "loss:", round(new_loss, 4)
        )


# Eğitim sonrası sonuç
final_prediction = neuron(x1, x2, w1, w2, b)
final_loss = loss(final_prediction, target)



print("w1: ", round(w1, 4))
print("prediction :", round(final_prediction, 3))
print("target: ", target)
print("Loss:", round(final_loss, 4))