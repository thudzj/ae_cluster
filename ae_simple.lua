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

model = model:cuda()
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

local d = 10
local mu_0 = torch:DoubleTensor(d):fill(0)

-- a point belong to which table
local belong = nil
-- a point is connected to which point
local connect = nil

-- record tables
local tables = {}
local n_tables = nil

-- a point connected by which points
local connected = {}


-- calculate the probability of a table
table_probability = function(h_i_mean, h_j_mean, h_ij_mean, nk_i, nk_j, nk_ij)
    -- maybe doesnot need to calculate this everytime: save h and lambda_nk and so on
    local rnt = math.exp( 
            nk_ij*nk_ij/2/(nk_ij+1)*(torch.norm(h_ij_mean)^2)
            - nk_i*nk_i/2/(nk_i+1)*(torch.norm(h_i_mean)^2)
            - nk_j*nk_j/2/(nk_j+1)*(torch.norm(h_j_mean)^2)) 
        *math.sqrt(nk_i+1) * math.sqrt(nk_j+1) / math.sqrt(nk_ij+1)
    if rnt == math.huge then
        print("!!!Attention: table's probability is inf now!!!")
    end
    return rnt
end

-- detach a connect and then generate a new table
split_table = function(table_id, new_table_id, point_id, new_table)
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

gibbs_sample = function(z, n_points)

    for i = 1, n_points do
        -- print(i)
        if i % 1000 == 0 then
            print(string.format("[%s], gibbs sampling, point: #%d, # of tables: %d", os.date("%c", os.time()), i, n_tables))
        end

        -- cache the proportion in the part 3 of equation 7
        local proportion = {}

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
            n_tables = n_tables + 1
        end
        -- 3. calculate the table_probability of the new table
        local nk_i = #tables[i]
        local h_i = torch.DoubleTensor(nk_i, d)
        for j = 1, nk_i do
            h_i[j] = z[tables[i][j]]
        end
        local h_i_mean = torch.mean(h_i, 1)

        assert(belong[i] == i, "i must be the first of the new table")
        -- sample the new c_i
        for j = 1, i - 1 do
            -- equation 7, part 2
            probability[j] = math.exp(-torch.norm(z[i] - z[j]))
            if belong[i] ~= belong[j] then
                -- if not in cache
                if proportion[belong[j]] == nil then
                    local nk_j = #tables[belong[j]]
                    local h_j = torch.DoubleTensor(nk_j, d)
                    for jj = 1, nk_j do
                        h_j[jj] = z[tables[belong[j]][jj]]
                    end
                    local h_j_mean = torch.mean(h_j, 1)
                    local h_ij_mean = torch.mean(torch.cat(h_i, h_j, 1), 1)
                    
                    proportion[belong[j]] = table_probability(h_i_mean, h_j_mean, h_ij_mean, nk_i, nk_j, nk_j+nk_i)
                    -- print (nk_i, nk_j, table_probabilitys[belong[j]], joint_table_probabilitys[belong[j]], table_probability_i)

                end
                
                -- equation 7, part 3
                probability[j] = probability[j] * proportion[belong[j]]
                
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
        -- print(i, target, sum_pro, probability[1])
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

        end
        
    end
end

cal_gradient = function(z, n_points)
    print(string.format("[%s], calculate gradient of DDCRP", os.date("%c", os.time())))
    local dz = torch.CudaTensor(z:size()):fill(0)
    local sum_f_ii = torch.CudaTensor(n_points):fill(0)
    -- set sum_f_ii to be: sum^i_1(distanc_{i, j})
    -- print(string.format("[%s], start precompute", os.date("%c", os.time())))
    for i = 1, n_points do
        sum_f_ii[i] = torch.sum(torch.exp(-torch.norm(z[i]:view(1,d):expandAs(z[{{1, i}}]) - z[{{1, i}}], 2, 2)))
    end
    local l1_loss = 0
    -- part 1 of d(l_1)
    -- print(string.format("[%s], start part 1 of d(l_1)", os.date("%c", os.time())))
    for i = 1, n_points do
        l1_loss = l1_loss + torch.norm(z[i] - z[connect[i]]) + math.log(sum_f_ii[i])
        if i ~= connect[i] then
            local tmp = (z[i] - z[connect[i]]) / torch.norm(z[i] - z[connect[i]])
            dz[i] = dz[i] + tmp
            dz[connect[i]] = dz[connect[i]] - tmp
        end
    end
    -- part 2 of d(l_1)
    -- print(string.format("[%s], start part 2 of d(l_1)", os.date("%c", os.time())))
    for i = 1, n_points do
        local tmp = z[i]:view(1,d):expandAs(z) - z
        local tmp_norm2 = torch.norm(tmp, 2, 2)
        if i > 1 then
            dz[i] = dz[i] - torch.sum(
                torch.cmul(
                    tmp[{{1, i-1}}],
                    torch.cdiv(
                        torch.exp(-tmp_norm2[{{1, i-1}}]), 
                        tmp_norm2[{{1, i-1}}]
                    ):expandAs(tmp[{{1, i-1}}])
                )
                /sum_f_ii[i], 
            1)
        end

        if i < n_points then
            dz[i] = dz[i] - torch.sum(
                torch.cmul(
                    tmp[{{i+1, n_points}}],
                    torch.cdiv(
                        torch.exp(-tmp_norm2[{{i+1, n_points}}]), 
                        torch.cmul(
                            tmp_norm2[{{i+1, n_points}}],
                            sum_f_ii[{{i+1, n_points}}]
                        )
                    ):expandAs(tmp[{{i+1, n_points}}])
                ), 
            1)
        end
        -- for j = 1, i - 1 do
        --     local tmp = torch.norm(z[i]-z[j])
        --     dz[i] = dz[i] - (z[i]-z[j])/tmp * math.exp(-tmp) / sum_f_ii[i]
        -- end
        -- for j = i + 1, n_points do
        --     local tmp = torch.norm(z[j]-z[i])
        --     dz[i] = dz[i] - (z[i]-z[j])/tmp * math.exp(-tmp) / sum_f_ii[j]
        -- end
    end
    -- d(l_2)
    -- print(string.format("[%s], start d(l_2)", os.date("%c", os.time())))
    local l2_loss = 0
    for table_id, table in pairs(tables) do
        local nk = #table
        if nk > 0 then
            print(string.format("[%s], table: %d, size: %d", os.date("%c", os.time()), table_id, nk))
            local h = torch.CudaTensor(nk, d)
            for i = 1, nk do
                h[i] = z[table[i]]
            end
            local h_mean = torch.mean(h, 1):view(1,d)
            local h_ = -(nk/(nk+1)*h_mean):expandAs(h) + h

            l2_loss = l2_loss + nk*nk/2/(nk+1)*(torch.norm(h_mean)^2) - 0.5*(torch.norm(h)^2) - 0.5*math.log(nk+1) - 0.5*nk*d*math.log(2*math.pi)
            
            for i = 1, nk do
                dz[table[i]] = h_[i]
            end
        end
    end
    return l1_loss, -l2_loss, dz
end

-- n_points is the size of dataset, delta is the weight of the gradient of log-likelihood function
train = function(n_epoches, sample_times, K, delta) 
    sgd_params.learningRate = 0.01
    sgd_params.weightDecay = 0
    sgd_params.momentum = 0.9
    sgd_params.learningRateDecay = 0
    local x, dl_dx = model:getParameters()
    local inference_net = nn.Sequential():cuda()
    inference_net:add(model.modules[1])
    inference_net:add(model.modules[2])
    inference_net:add(model.modules[3])
    inference_net:add(model.modules[4])
    inference_net:add(model.modules[5])
    inference_net:add(model.modules[6])
    inference_net:add(model.modules[7])
    inference_net:add(model.modules[8])

    -- initialize
    local n_points = trainset.size
    belong = torch.range(1, n_points):int()
    connect = torch.range(1, n_points):int()

    n_tables = n_points
    for i = 1, n_points do
        tables[i] = {}
        table.insert(tables[i], i)
    end

    for i = 1, n_points do
        connected[i] = {}
        table.insert(connected[i], i)
    end
    
    -- eval function
    local feval = function(x_new)
        if x ~= x_new then x:copy(x_new) end
        dl_dx:zero()

        -- By default, the losses are averaged over observations for each minibatch. 
        -- However, if the field sizeAverage is set to false, the losses are instead summed
        local loss = criterion:forward(model:forward(trainset.data), trainset.data)
        print(string.format("[%s], auto-encoder loss: %4f", os.date("%c", os.time()), loss))
        model:backward(trainset.data, criterion:backward(model.output, trainset.data))

        local z = model.modules[8].output:double() --z is the hidden features
        
        for ite = 1, sample_times do
            gibbs_sample(z, n_points)
        end

        for ite = 1, K do
            gibbs_sample(z, n_points)
            l1_loss, l2_loss, dz = cal_gradient(model.modules[8].output, n_points)
            dz = dz:cuda() * delta / K
            loss = loss + (l1_loss + l2_loss) * delta / K
            print(ite, l1_loss, l2_loss)

            -- local ttmp = dl_dx:clone()
            inference_net:backward(trainset.data, dz)
            -- local tttmp = dl_dx
            -- print(torch.all(torch.eq(ttmp, tttmp)))
        end

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

print(string.format("[%s], start training...", os.date("%c", os.time())))
train(100, 3, 3, 0.0000015)


-- linear = model.modules[4]

-- vec = torch.zeros(layer_size)
-- vec[1] = 1

-- translate = nn.Sequential()
-- translate:add(linear)
-- translate:add(nn.Reshape(28, 28))

-- itorch.image(translate:forward(vec))

-- basis = torch.eye(layer_size)

-- itorch.image(translate:forward(basis))