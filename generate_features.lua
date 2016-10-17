require 'torch'
require 'nn'
require 'optim'
require 'cunn'
require 'cutorch'
require 'itorch'

print(string.format("GPU number: %d", cutorch.getDeviceCount()))
cutorch.setDevice(3)
print(string.format("Using GPU %d", cutorch.getDevice()))

mnist = require 'mnist'
fullset = mnist.traindataset()
testset = mnist.testdataset()
trainset = {
    size = 1000,
    data = fullset.data[{{1,1000}}]:double() / 256,
    label = fullset.label[{{1,1000}}]
}
trainset.data = trainset.data:cuda()

model_name = "models/mnist_ae"
model = torch.load(model_name.."_100")

model:forward(trainset.data)

torch.save('data/feature_100.t7', model.modules[8].output:double())