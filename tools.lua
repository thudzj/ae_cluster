require 'torch'
require 'nn'
require 'optim'
require 'cunn'
require 'cutorch'


-- layer-wise pretrain the auto-encoder
layerwise_pretrain = function(model, criterion, batch_size, sgd_params, trainset)
    batch_size = batch_size or 200
    local base_lr = 0.1
    local lr_decay = 20000

    local pair={}
    pair[1] = nn.Sequential()
    pair[1]:add(nn.Dropout(0.2))
    pair[1]:add(model.modules[1])
    pair[1]:add(model.modules[2])
    pair[1]:add(model.modules[3])
    pair[1]:add(nn.Dropout(0.2))
    pair[1]:add(model.modules[15])
    pair[1]:add(model.modules[16])
    pair[1]:add(model.modules[17])

    pair[2] = nn.Sequential()
    pair[2]:add(nn.Dropout(0.2))
    pair[2]:add(model.modules[4])
    pair[2]:add(model.modules[5])
    pair[2]:add(nn.Dropout(0.2))
    pair[2]:add(model.modules[13])
    pair[2]:add(model.modules[14])

    pair[3] = nn.Sequential()
    pair[3]:add(nn.Dropout(0.2))
    pair[3]:add(model.modules[6])
    pair[3]:add(model.modules[7])
    pair[3]:add(nn.Dropout(0.2))
    pair[3]:add(model.modules[11])
    pair[3]:add(model.modules[12])

    pair[4] = nn.Sequential()
    pair[4]:add(nn.Dropout(0.2))
    pair[4]:add(model.modules[8])
    -- pair[4]:add(nn.Dropout(0.2))
    pair[4]:add(model.modules[9])
    pair[4]:add(model.modules[10])

    for i = 1,4 do
        pair[i] = pair[i]:cuda()
    end

    local extractor = {}
    extractor[2] = 3
    extractor[3] = 5
    extractor[4] = 7

    local feature_size = {}
    feature_size[2] = 500
    feature_size[3] = 500
    feature_size[4] = 2000

    for i = 1,4 do
        print("Training a pair...")
        print(pair[i])
        local x, dl_dx = pair[i]:getParameters()
        local iters = 0
        local n_iters = 40000
        local featureset = nil
        local state = {}
        state.evalCounter = 0
        state.dfdx = nil
        if i == 1 then
            -- goto continue
            featureset = torch.Tensor(trainset.size, 28, 28):cuda()
            for j = 1, trainset.size do
                featureset[j] = trainset.data[j]
            end
        else
            featureset = torch.Tensor(trainset.size, feature_size[i]):cuda()
            for j = 1, trainset.size do
                model:forward(trainset.data[j])
                featureset[j] = model.modules[extractor[i]].output
            end
        end
        while 1 do
            local shuffle = torch.randperm(trainset.size):cuda()
            for t = 1,trainset.size,batch_size do
                local size = math.min(t + batch_size - 1, trainset.size) - t
                local inputs = nil
                if i == 1 then
                    inputs = torch.Tensor(size, 28, 28):cuda()
                else
                    inputs = torch.Tensor(size, feature_size[i]):cuda()
                end
                for j = 1,size do
                    inputs[j] = featureset[shuffle[j+t]]
                end
                
                local feval = function(x_new)
                    -- reset data
                    if x ~= x_new then x:copy(x_new) end
                    dl_dx:zero()

                    -- perform mini-batch gradient descent
                    local loss = criterion:forward(pair[i]:forward(inputs), inputs)
                    pair[i]:backward(inputs, criterion:backward(pair[i].output, inputs):cuda())
                    return loss, dl_dx
                end
                sgd_params.learningRate = base_lr * math.pow(0.1, iters / lr_decay)

                _, fs = optim.sgd(feval, x, sgd_params, state)
                
                iters = iters + 1
                if iters % 500 == 0 then
                    print(string.format('Layer-wise pretraining pair: %d, epoch: %d, lr: %.4f, current loss: %4f', 
                        i, iters, sgd_params.learningRate, fs[1]))
                end
                if iters == n_iters then
                    break
                end
            end
            if iters == n_iters then
                break
            end
        end
        ::continue::
    end
end


-- end to end pretrain the auto-encoder
e2e_finetune = function(model, criterion, batch_size, sgd_params, trainset)
    local x, dl_dx = model:getParameters()
    local iters = 0
    local n_iters = 100000
    local state = {}
    state.evalCounter = 0
    state.dfdx = nil
    batch_size = batch_size or 200
    local base_lr = 0.1
    local lr_decay = 20000
    while 1 do
        local shuffle = torch.randperm(trainset.size):cuda()
        
        for t = 1,trainset.size,batch_size do
            -- setup inputs for this mini-batch
            -- no need to setup targets, since they are the same
            local size = math.min(t + batch_size - 1, trainset.size) - t
            local inputs = torch.Tensor(size, 28, 28):cuda()
            for i = 1,size do
                inputs[i] = trainset.data[shuffle[i+t]]
            end
            
            local feval = function(x_new)
                -- reset data
                if x ~= x_new then x:copy(x_new) end
                dl_dx:zero()

                -- perform mini-batch gradient descent
                local loss = criterion:forward(model:forward(inputs), inputs)
                model:backward(inputs, criterion:backward(model.output, inputs))

                return loss, dl_dx
            end

            sgd_params.learningRate = base_lr * math.pow(0.1, iters / lr_decay)
            _, fs = optim.sgd(feval, x, sgd_params, state)
            
            iters = iters + 1
            if iters % 500 == 0 then
                print(string.format('End to end finetuning, epoch: %d, lr: %.4f, current loss: %4f', 
                    iters, sgd_params.learningRate, fs[1]))
            end
            if iters == n_iters then
                break
            end
        end
        if iters == n_iters then
            break
        end
    end
end