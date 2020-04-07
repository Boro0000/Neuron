import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  f = sigmoid(x)
  return f * (1 - f)

def mse_loss(y, y_pred):
  return ((y - y_pred) ** 2).mean()

class Neuron:
  def __init__(self):
    self.b1 = np.random.random()
    self.b2 = np.random.random()
    self.b3 = np.random.random()
    self.w0 = np.random.random()
    self.w1 = np.random.random()
    self.w2 = np.random.random()
    self.w3 = np.random.random()
    self.w4 = np.random.random()
    self.w5 = np.random.random()
    print("B: B1= %.10f B2= %.10f B3= %.10f" % (self.b1, self.b2,self.b3))
    print(" ")
    print("W: W0= %.10f W1= %.10f W2= %.10f W3= %.10f W4= %.10f W5= %.10f" % (self.w0, self.w1,self.w2,self.w3,self.w4,self.w5))
    print(" ")
  def feedforward(self, x):
    h0 = sigmoid(self.w0 * x[0] + self.w1 * x[1] + self.b1)
    h1 = sigmoid(self.w2 * x[0] + self.w3 * x[1] + self.b2)
    o0 = sigmoid(self.w4 * h0 + self.w5 * h1 + self.b3)
    return o0

  def train(self, data, all_y,learn_rate,epochs):
    for epoch in range(epochs+1):
      for x, y in zip(data, all_y):
        sumh0 = self.w0 * x[0] + self.w1 * x[1] + self.b1
        h0 = sigmoid(sumh0)
        sumh1 = self.w2 * x[0] + self.w3 * x[1] + self.b2
        h1 = sigmoid(sumh1)
        sumo0 = self.w4 * h0 + self.w5 * h1 + self.b3
        o0 = sigmoid(sumo0)
        y_pred = o0
        dypred = -2 * (y- y_pred)
        #Neuron o0
        dw4 = h0 * deriv_sigmoid(sumo0)
        dw5 = h1 * deriv_sigmoid(sumo0)
        db3 = deriv_sigmoid(sumo0)
        dh0 = self.w4 * deriv_sigmoid(sumo0)
        dh1 = self.w5 * deriv_sigmoid(sumo0)
        #Neuron h0
        dw0 = x[0] * deriv_sigmoid(sumh0)
        dw1 = x[1] * deriv_sigmoid(sumh0)
        db1 = deriv_sigmoid(sumh0)
        #Neuron h1
        dw2 = x[0] * deriv_sigmoid(sumh1)
        dw3 = x[1] * deriv_sigmoid(sumh1)
        db2 = deriv_sigmoid(sumh1)
        #Neuron h0
        self.w0 -= learn_rate * dypred * dh0 * dw0
        self.w1 -= learn_rate * dypred * dh0 * dw1
        self.b1 -= learn_rate * dypred * dh0 * db1
        #Neuron h1
        self.w2 -= learn_rate * dypred * dh1 * dw2
        self.w3 -= learn_rate * dypred * dh1 * dw3
        self.b2 -= learn_rate * dypred * dh1 * db2
        #Neuron o0
        self.w4 -= learn_rate * dypred * dw4
        self.w5 -= learn_rate * dypred * dw5
        self.b3 -= learn_rate * dypred * db3
      if epoch % 2 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y, y_preds)
        print("Epoch %d strata: %.10f" % (epoch, loss))

data = np.array([[0, 0],[0, 1],[1,0],[1, 1],])
all_y = np.array([0,1,1,1,])
network = Neuron()
network.train(data, all_y,0.1,1000)