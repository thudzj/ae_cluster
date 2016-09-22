require 'torch'
require 'nn'
require 'optim'
require 'cunn'
require 'cutorch'
require 'cephes'

print(string.format("GPU number: %d", cutorch.getDeviceCount()))
cutorch.setDevice(2)
print(string.format("Using GPU %d", cutorch.getDevice()))

--require 'itorch'
mnist = require 'mnist'

fullset = mnist.traindataset()
testset = mnist.testdataset()

trainset = {
    size = 10000, --70000,
    data = fullset.data[{{1,10000}}]:double()/256,--torch.cat(fullset.data[{{1,60000}}], testset.data[{{1,10000}}], 1):double() / 256,
    label = fullset.label[{{1,10000}}]--torch.cat(fullset.label[{{1,60000}}], testset.label[{{1,10000}}],1)
}

trainset.data = trainset.data:cuda()
trainset.label = trainset.label:cuda()

-- validationset = {
--     size = 10000,
--     data = fullset.data[{{50001,60000}}]:double() / 256,
--     label = fullset.label[{{50001,60000}}]
-- }

model_name = "mnist_ae"

model = nn.Sequential()
model:add(nn.Reshape(28*28))
model:add(nn.Linear(28*28, 500))
model:add(nn.ReLU())
model:add(nn.Linear(500, 500))
model:add(nn.ReLU())
model:add(nn.Linear(500, 2000))
model:add(nn.ReLU())
model:add(nn.Linear(2000, 10))
--model:add(nn.ReLU())
model:add(nn.Linear(10, 2000))
model:add(nn.ReLU())
model:add(nn.Linear(2000, 500))
model:add(nn.ReLU())
model:add(nn.Linear(500, 500))
model:add(nn.ReLU())
model:add(nn.Linear(500, 28*28))
--model:add(nn.ReLU())
model:add(nn.Reshape(28, 28))

for i=1,model:size() do
    --print (model.modules[i])
    if model.modules[i].weight then
        model.modules[i].weight = torch.randn(model.modules[i].weight:size()) * 0.01
    end
end

inference_net = nn.Sequential()
inference_net:add(model.modules[1])
inference_net:add(model.modules[2])
inference_net:add(model.modules[3])
inference_net:add(model.modules[4])
inference_net:add(model.modules[5])
inference_net:add(model.modules[6])
inference_net:add(model.modules[7])
inference_net:add(model.modules[8])

model = model:cuda()
inference_net = inference_net:cuda()

--torch.save(model_name, model)

pair={}
pair[1] = nn.Sequential()
pair[1]:add(model.modules[1])
pair[1]:add(model.modules[2])
pair[1]:add(model.modules[3])
pair[1]:add(nn.Dropout(0.2))
pair[1]:add(model.modules[15])
pair[1]:add(model.modules[16])

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
pair[4]:add(model.modules[9])
pair[4]:add(model.modules[10])

for i = 1,4 do
    pair[i] = pair[i]:cuda()
end

extractor = {}
extractor[2] = 3
extractor[3] = 5
extractor[4] = 7

feature_size = {}
feature_size[2] = 500
feature_size[3] = 500
feature_size[4] = 2000

batch_size = 256
base_lr = 0.1
lr_decay = 20000

sgd_params = {
   learningRate = 0.1,
   learningRateDecay = 0,
   weightDecay = 0,
   momentum = 0.9
}

criterion = nn.MSECriterion()
criterion = criterion:cuda()

-- layer-wise pretrain the auto-encoder
layerwise_pretrain = function(batch_size)
    batch_size = batch_size or 200
    for i = 1,4 do
        print("Training a pair...")
        print(pair[i])
        local x, dl_dx = pair[i]:getParameters()
        x = x:cuda()
        dl_dx = dl_dx:cuda()
        local iters = 0
        local n_iters = 50000
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
e2e_finetune = function(batch_size)
    local x, dl_dx = model:getParameters()
    local iters = 0
    local n_iters = 100000
    local state = {}
    state.evalCounter = 0
    state.dfdx = nil
    batch_size = batch_size or 200
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

-- n_points is the size of dataset, delta is the weight of the gradient of log-likelihood function
train = function(n_epoches, sample_times, n_points, delta) 
    sgd_params.learningRate = 0.01
    sgd_params.weightDecay = 0
    sgd_params.momentum = 0.9
    sgd_params.learningRateDecay = 0
    local x, dl_dx = model:getParameters()
    -- local distance = torch.DoubleTensor(n_points, n_points)
    local d = 10
    local kappa_0 = 1
    local nu_0 = 10 -- not sure
    local mu_0 = torch:DoubleTensor(d):fill(0)
    local Lambda_0 = torch.eye(d)
    local Lambda_0_det_pow_nu0_div_2 = 1

    -- a point belong to which table
    local belong = torch.range(1, n_points):int()
    -- a point is connected to which point
    local connect = torch.range(1, n_points):int()
    -- cache the tables' probability
    local table_probabilitys = torch.DoubleTensor(n_points):fill(0)

    -- record tables
    local tables = {}
    local n_tables = n_points
    for i = 1, n_points do
        tables[i] = {}
        table.insert(tables[i], i)
    end

    -- a point connected by which points
    local connected = {}
    for i = 1, n_points do
        connected[i] = {}
        table.insert(connected[i], i)
    end

    -- cache the multivariate_gamma function
    local multivariate_gamma_table = torch.DoubleTensor(n_points)

    local multi_gamma = function(xx)
        -- https://en.wikipedia.org/wiki/Multivariate_gamma_function
        local result = 1 -- math.pow(math.pi, d * (d - 1) / 4)
        for j = 1, d do
            result = result * cephes.gamma(xx + (1-j) / 2) / cephes.gamma((nu_0+1-j)/2)
        end
        return result
    end

    -- i > 54, is inf
    for i = 1, 54 do
        multivariate_gamma_table[i] = multi_gamma((nu_0 + i)/ 2)
    end

    -- calculate the probability of a table
    local table_probability = function(h)
        -- maybe doesnot need to calculate this everytime: save h and lambda_nk and so on
        table_size = h:size(1)
        print(table_size)
        local kappa_nk = kappa_0 + table_size
        local nu_nk = nu_0 + table_size
        local h_mean = torch.mean(h, 1):view(1,d)
        local h_ = h - h_mean:expandAs(h)
        local Lambda_nk = Lambda_0 + (h_:transpose(1,2))*h_ + (h_mean:transpose(1,2)) * h_mean 
            * table_size * kappa_0 / (kappa_0 + table_size)
        local Lambda_nk_e = torch.symeig(Lambda_nk, 'N') -- determinant = PI_i(eigenvalue_i)
        local result = math.pow(1/math.pi, table_size*d/2) * math.pow(kappa_0/kappa_nk, d/2) 
                        * multivariate_gamma_table[table_size] * Lambda_0_det_pow_nu0_div_2
        return result / torch.prod(Lambda_nk_e)
    end

    -- detach a connect and then generate a new table
    local split_table = function(table_id, new_table_id, point_id, new_table)
        table.insert(new_table, point_id)
        belong[point_id] = new_table_id
        for j = 1, #tables[table_id] do
            if tables[table_id][j] == point_id then
                table.remove(tables[table_id], j)
                break
            end
        end
        for i = 1, #connected[point_id] do
            if connected[point_id][i] ~= point_id then
                split_table(table_id, new_table_id, connected[point_id][i], new_table)
            end
        end
    end

    local gibbs_sample = function( z )

        for i = 1, n_points do
            if i % 1 == 0 then
                print(string.format("[%s], gibbs sampling, point: #%d, # of tables: %d", os.date("%c", os.time()), i, n_tables))
            end

            -- cache the table_probability of two joint tables
            local joint_table_probabilitys = torch.DoubleTensor(i):fill(0)

            -- multinomial distribution
            local probability = torch.DoubleTensor(i)
            -- patition Z
            local sum_pro = 0.0

            -- detach the table
            -- 1. remove the connect of connected[connect[i]]
            for j = 1, #connected[connect[i]] do
                if connected[connect[i]][j] == i then
                    table.remove(connected[connect[i]], j)
                    break
                end
            end
            -- 2. if c_i is not self-connected, generative a new table
            if connect[i] ~= i then
                tables[i] = {}
                split_table(belong[i], i, i, tables[i])
                -- clear the probability cache of table belong[i]
                table_probabilitys[belong[i]] = 0
            end
            -- 3. calculate the table_probability of the new table
            local nk_i = #tables[i]
            local h_i = torch.DoubleTensor(nk_i, d)
            for j = 1, nk_i do
                h_i[j] = z[tables[i][j]]
            end
            table_probabilitys[i] = table_probability(h_i)


            assert(belong[i] == i, "i must be the first of the new table")
            -- sample the new c_i
            for j = 1, i - 1 do
                -- equation 7, part 2
                probability[j] = math.exp(-torch.norm(z[i] - z[j]))
                if belong[i] ~= belong[j] then
                    -- if not in cache
                    if table_probabilitys[belong[j]] == 0 or joint_table_probabilitys[belong[j]] == 0 then
                        local nk_j = #tables[belong[j]]
                        local h_j = torch.DoubleTensor(nk_j, d)
                        for jj = 1, nk_j do
                            h_j[jj] = z[tables[belong[j]][jj]]
                        end
                        if table_probabilitys[belong[j]] == 0 then
                            table_probabilitys[belong[j]] = table_probability(h_j)
                        end
                        if joint_table_probabilitys[belong[j]] == 0 then
                            joint_table_probabilitys[belong[j]] = table_probability(torch.cat(h_i, h_j, 1))
                        end

                        -- print (nk_i, nk_j, table_probabilitys[belong[j]], joint_table_probabilitys[belong[j]], table_probability_i)

                    end
                    
                    -- equation 7, part 3
                    probability[j] = probability[j] * joint_table_probabilitys[belong[j]] / table_probabilitys[belong[j]] / table_probabilitys[i]
                    
                end
                sum_pro = sum_pro + probability[j]
            end

            -- equation 7, part 1
            probability[i] = 1
            sum_pro = sum_pro + 1

            -- gibbs sampling
            local U = torch.uniform(0, sum_pro)
            local u = 0.0
            local target = 0
            for j = 1, i do
                u = u + probability[j]
                if u > U then
                    target = j
                    break
                end
            end

            -- assign the new connect
            assert(target > 0, "Target below zero is invalid.")
            connect[i] = target
            table.insert(connected[target], i)

            -- connect the new c_i and then update the tables and table_probabulitys
            if i ~= belong[target] then
                assert(belong[target] < belong[i], "The new c_i must be connected to a former point")
                for j = 1, #tables[i] do
                    table.insert(tables[belong[target]], tables[i][j])
                    belong[tables[i][j]] = belong[target]
                end
                n_tables = n_tables - 1
                tables[i] = {}
                table_probabilitys[i] = 0
                table_probabilitys[belong[target]] = joint_table_probabilitys[belong[target]]
            end
            
        end
    end

    local dz = nil
    local cal_gradient = function( z )
        dz = torch.DoubleTensor(z:size()):fill(0)
        local sum_f_ii = torch.DoubleTensor(n_points):fill(0)
        -- set sum_f_ii to be: sum^i_1(distanc_{i, j})
        for i = 1, n_points do
            for j = 1, i do
                sum_f_ii[i] = sum_f_ii[i] + math.exp(-torch.norm(z[i] - z[j]))
            end
        end
        -- part 1 of d(l_1)
        for i = 1, n_points do
            if i ~= connect[i] then
                local tmp = (z[i] - z[connect[i]]) / torch.norm(z[i] - z[connect[i]])
                dz[i] = dz[i] - tmp
                dz[connect[i]] = dz[connect[i]] + tmp
            end
        end
        -- part 2 of d(l_1)
        for i = 1, n_points do
            for j = 1, i - 1 do
                local tmp = torch.norm(z[i]-z[j])
                dz[i] = dz[i] + (z[i]-z[j])/tmp * math.exp(-tmp) / sum_f_ii[i]
            end
            for j = i + 1, n_points do
                local tmp = torch.norm(z[j]-z[i])
                dz[i] = dz[i] + (z[i]-z[j])/tmp * math.exp(-tmp) / sum_f_ii[j]
            end
        end
        -- d(l_2)
        for table_id, table in pairs(tables) do
            local nk = #table
            local h = torch.DoubleTensor(nk, d)
            for i = 1, nk do
                h[i] = z[table[i]]
            end
            local h_mean = torch.mean(h, 1):view(1,d)
            local h_ = h - h_mean:expandAs(h)
            local Lambda_nk_inv = torch.inverse(Lambda_0 + (h_:transpose(1,2))*h_ + 
                    (h_mean:transpose(1,2)) * h_mean*nk*kappa_0/(kappa_0+nk))
            local dh = torch.DoubleTensor(d, d)
            for i = 1, nk do
                for j = 1, d do
                    dh:fill(0)
                    -- combine equation 15 and 16
                    local tmp = h[i] - h_mean + kappa_0 / (kappa_0 + nk) * (h_mean - mu_0)
                    dh[j] = dh[j] + tmp
                    dh[{{}, j}] = dh[{{}, j}] + tmp
                    dz[table[i]][j] = dz[table[i]][j] - (nu_0 + nk) / 2 * torch.trace(Lambda_nk_inv * dh)
                end
            end
        end
    end

    local feval = function(x_new)
        if x ~= x_new then x:copy(x_new) end
        dl_dx:zero()

        -- By default, the losses are averaged over observations for each minibatch. 
        -- However, if the field sizeAverage is set to false, the losses are instead summed
        local loss = criterion:forward(model:forward(trainset.data), trainset.data)
        local z = model.modules[8].output:double() --z is the hidden features
        
        -- reset the table_probabilitys because z is changed
        table_probabilitys:fill(0)
        for ite = 1, sample_times do
            gibbs_sample(z)
        end

        cal_gradient(z)
        dz = dz:cuda() * delta

        model:backward(trainset.data, criterion:backward(model.output, trainset.data))
        inference_net:backward(trainset.data, dz)

        return loss, dl_dx
    end

    for epoch = 1, n_epoches do
        _, fs = optim.sgd(feval, x, sgd_params)
        print (string.format('Jointly training, epoch: %d, current loss: %4f', epoch, fs[1]))
    end
end

-- eval function of the auto-encoder
eval = function(dataset, batch_size)
    local loss = 0
    batch_size = batch_size or 200
    
    for i = 1,dataset.size,batch_size do
        local size = math.min(i + batch_size - 1, dataset.size) - i
        local inputs = dataset.data[{{i,i+size-1}}]
        local outputs = model:forward(inputs)
        loss = criterion:forward(model:forward(inputs), inputs)
        print(string.format("Eval loss on test set, iter: %d, loss: %4f", i, loss))
    end

end

if not path.exists(model_name) then
    layerwise_pretrain(batch_size)
    e2e_finetune(batch_size)
    torch.save(model_name, model)
else 
    model = torch.load(model_name)
end

-- print("Loss on test set of pretrained model:")
-- testset.data = testset.data:double() / 256
-- testset.data = testset.data:cuda()
-- eval(testset, batch_size)

print("Start training...")
train(100, 10, trainset.size, 0.1)


-- linear = model.modules[4]

-- vec = torch.zeros(layer_size)
-- vec[1] = 1

-- translate = nn.Sequential()
-- translate:add(linear)
-- translate:add(nn.Reshape(28, 28))

-- itorch.image(translate:forward(vec))

-- basis = torch.eye(layer_size)

-- itorch.image(translate:forward(basis))