require 'torch'
require 'nn'
require 'optim'
require 'cunn'
require 'cutorch'
require 'itorch'
require 'dpnn'
require 'tools'

print(string.format("GPU number: %d", cutorch.getDeviceCount()))
cutorch.setDevice(3)
print(string.format("Using GPU %d", cutorch.getDevice()))

mnist = require 'mnist'
fullset = mnist.traindataset()
testset = mnist.testdataset()

sgd_params = {
   learningRate = 0.1,
   learningRateDecay = 0,
   weightDecay = 0.01,
   momentum = 0.9
}
print(sgd_params)

criterion = nn.MSECriterion()
criterion = criterion:cuda()

model_name = "models/mnist_ae"
tag = 2
load_name = model_name .. string.format("_e_%d", tag)
if not path.exists(load_name) then
    local trainset = {
        size = 70000,
        data = torch.cat(fullset.data[{{1,60000}}], testset.data[{{1,10000}}], 1):double() / 256,
        label = torch.cat(fullset.label[{{1,60000}}], testset.label[{{1,10000}}],1)
    }
    trainset.data = trainset.data:cuda()

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
        if model.modules[i].weight then
            model.modules[i].weight = torch.randn(model.modules[i].weight:size()) * 0.01
        end
    end
    model = model:cuda()
    layerwise_pretrain(model, criterion, 256, sgd_params, trainset)

    noiser = nn.WhiteNoise(0, 0.1):cuda()
    model:insert(noiser, 1)
    e2e_finetune(model, criterion, 256, sgd_params, trainset)
    torch.save(load_name, model)
else
    model = torch.load(load_name)
end

-- hidden feature size
local d = 10
-- a point belong to which table
local belong = nil
-- a point is connected to which point
local connect = nil
-- record tables
local tables = {}
local n_tables = nil
-- a point connected by which points
local connected = {}
-- alpha
local alpha = 1
local lambda = 1 -- 0.5

-- calculate the probability of a table
table_probability = function(h_i_mean, h_j_mean, h_ij_mean, nk_i_in, nk_j_in, nk_ij_in)
    nk_ij = nk_ij_in / lambda
    nk_i = nk_i_in / lambda
    nk_j = nk_j_in / lambda
    local rnt = math.exp( 
            nk_ij*nk_ij/2/(nk_ij+1)*h_ij_mean
            - nk_i*nk_i/2/(nk_i+1)*h_i_mean
            - nk_j*nk_j/2/(nk_j+1)*h_j_mean)
        * math.pow(math.sqrt(nk_i+1) * math.sqrt(nk_j+1) / math.sqrt(nk_ij+1), d)
    if rnt == math.huge then
        print("!!!Attention: table's probability is inf now!!!")
    end
    -- if rnt == 0 then
    --     print("!!!Attention: table's probability is 0 now!!!")
    --     print(h_i_mean, h_j_mean, h_ij_mean, nk_i, nk_j, nk_ij)
    -- end
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
        local h_i_mean = torch.norm(torch.mean(h_i, 1))^2

        assert(belong[i] == i, "i must be the first of the new table")
        -- sample the new c_i
        for j = 1, i - 1 do
            -- equation 7, part 2
            probability[j] = math.exp(-torch.norm(z[i] - z[j])/alpha)
            assert(belong[i] ~= belong[j], "belong[i] ~= belong[j]")
            -- if not in cache
            if proportion[belong[j]] == nil then
                local nk_j = #tables[belong[j]]
                local h_j = torch.DoubleTensor(nk_j, d)
                for jj = 1, nk_j do
                    h_j[jj] = z[tables[belong[j]][jj]]
                end
                local h_j_mean = torch.norm(torch.mean(h_j, 1))^2
                local h_ij_mean = torch.norm(torch.mean(torch.cat(h_i, h_j, 1), 1))^2
                
                proportion[belong[j]] = table_probability(h_i_mean, h_j_mean, h_ij_mean, nk_i, nk_j, nk_j+nk_i)
            end
            
            -- equation 7, part 3
            probability[j] = probability[j] * proportion[belong[j]]
                
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

        -- local cnt = 0
        -- local cnt2 = 0
        -- for table_id, table in pairs(tables) do
        --     cnt2 = cnt2 + #table
        --     if #table > 0 then
        --         cnt = cnt + 1
        --     end
        -- end

        -- if i % 10000 == 0 then
        --     print(string.format("   [%s], gibbs sampling, point: #%d, # of tables: %d, cnt: %d, cnt2: %d", os.date("%c", os.time()), i, n_tables, cnt, cnt2))
        -- end
        
    end

    print(string.format("   [%s], gibbs sampling, # of points: %d, # of tables: %d", os.date("%c", os.time()), n_points, n_tables))
end

cal_gradient = function(z, n_points)
    -- print(string.format("   [%s], calculating gradient of DDCRP", os.date("%c", os.time())))
    local dz = torch.CudaTensor(z:size()):fill(0)
    local sum_f_ii = torch.CudaTensor(n_points):fill(0)
    -- set sum_f_ii to be: sum^i_1(distanc_{i, j})
    -- print(string.format("[%s], start precompute", os.date("%c", os.time())))
    for i = 1, n_points do
        sum_f_ii[i] = torch.sum(torch.exp(-torch.norm(z[i]:view(1,d):expandAs(z[{{1, i}}]) - z[{{1, i}}], 2, 2)/alpha))
    end
    local l1_loss = 0
    -- part 1 of d(l_1)
    -- print(string.format("[%s], start part 1 of d(l_1)", os.date("%c", os.time())))
    for i = 1, n_points do
        l1_loss = l1_loss + torch.norm(z[i] - z[connect[i]])/alpha + math.log(sum_f_ii[i])
        if i ~= connect[i] then
            local tmp = (z[i] - z[connect[i]]) / torch.norm(z[i] - z[connect[i]])
            dz[i] = dz[i] + tmp/alpha
            dz[connect[i]] = dz[connect[i]] - tmp/alpha
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
                        torch.exp(-tmp_norm2[{{1, i-1}}]/alpha), 
                        alpha*tmp_norm2[{{1, i-1}}]
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
                        torch.exp(-tmp_norm2[{{i+1, n_points}}]/alpha), 
                        alpha*torch.cmul(
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
            -- print(string.format("[%s], table: %d, size: %d", os.date("%c", os.time()), table_id, nk))
            local h = torch.CudaTensor(nk, d)
            for i = 1, nk do
                h[i] = z[table[i]]
            end
            local h_mean = torch.mean(h, 1):view(1,d)
            local h_ = (h/lambda-(nk/(nk*lambda+lambda*lambda)*h_mean):expandAs(h))

            l2_loss = l2_loss - (nk/lambda)*(nk/lambda)/2/(nk/lambda+1)*(torch.norm(h_mean)^2) + (torch.norm(h)^2)/2/lambda + 0.5*d*math.log(nk/lambda+1) + 0.5*nk*d*math.log(2*math.pi*lambda)
            
            for i = 1, nk do
                dz[table[i]] = h_[i]
            end
        end
    end
    return l1_loss, l2_loss, dz
end

visualize = function(labels, z, epoch)
    local x = torch.Tensor(labels:size())
    local y = torch.Tensor(labels:size())

    local cnt = 0
    local table_heads = nil
    local maps = {}
    for table_id, table in pairs(tables) do
        if #table > 0 then
            cnt = cnt + 1
            maps[cnt] = table_id
            if table_heads == nil then
                table_heads = z[table_id]:view(1, d)
            else
                table_heads = torch.cat(table_heads, z[table_id]:view(1,d), 1)
            end
        end
    end

    local mean = torch.mean(table_heads, 1) -- 1 x n
    local m = table_heads:size(1)
    local Xm = table_heads - torch.ones(m, 1) * mean
    Xm:div(math.sqrt(m - 1))
    vecs,s_,_ = torch.svd(Xm:t())
    X_hat = (table_heads - torch.ones(m,1) * mean) * vecs[{ {},{1, 2} }]

    for i = 1, #maps do
        x[maps[i]] = X_hat[i][1]
        y[maps[i]] = X_hat[i][2]
    end

    for table_id, table in pairs(tables) do
        for i = 1, #table do
            if table[i] ~= table_id then
                x[table[i]] = torch.randn(1)*0.1 + x[table_id]
                y[table[i]] = torch.randn(1)*0.1 + y[table_id]
            end
        end
    end

    itorch.Plot():gscatter(x,y,labels):title(string.format('Epoch %d clustering result(clusters: %d)', epoch, n_tables)):save(string.format('visualization/visualization_%d.html', epoch))
end

-- n_points is the size of dataset, delta is the weight of the gradient of log-likelihood function
train = function(model, inference_net, trainset, z_index, n_points, n_epoches, sample_times, K, delta, lr_d, lr_a, lr_decay) 
    -- sgd parameters
    sgd_params.learningRate = 0
    sgd_params.weightDecay = 0.0001
    sgd_params.momentum = 0.9
    sgd_params.learningRateDecay = 0

    print(alpha, lambda, lr_d, lr_a, lr_decay, sgd_params.weightDecay, sgd_params.momentum)    

    local x, dl_dx = model:getParameters()
    local state_ddcrp = {}
    state_ddcrp.evalCounter = 0
    state_ddcrp.dfdx = nil
    local state_ae = {}
    state_ae.evalCounter = 0
    state_ae.dfdx = nil

    
    
    -- eval function for sgd optimizer
    local feval_ddcrp = function(x_new)
        if x ~= x_new then x:copy(x_new) end
        dl_dx:zero()

        inference_net:forward(trainset.data)
        local z = inference_net.output:double() --z is the hidden features
        local loss = 0
        
        for ite = 1, sample_times do
            gibbs_sample(z, n_points)
        end

        for ite = 1, K do
            gibbs_sample(z, n_points)
            l1_loss, l2_loss, dz = cal_gradient(inference_net.output, n_points)
            dz = dz:cuda() / K / n_points
            inference_net:backward(trainset.data, dz)

            loss = loss + (l1_loss + l2_loss) / K / n_points
            print(string.format("   [%s], optimizing DDCRP, loss: %f, loss1: %f, loss2: %f", os.date("%c", os.time()), loss, l1_loss / K / n_points, l2_loss / K / n_points))
        end

        return loss, dl_dx
    end

    local feval_ae = function(x_new)
        -- reset data
        if x ~= x_new then x:copy(x_new) end
        dl_dx:zero()

        -- perform mini-batch gradient descent
        local loss = criterion:forward(model:forward(trainset.data), trainset.data)
        model:backward(trainset.data, criterion:backward(model.output, trainset.data))

        return loss, dl_dx
    end

    local h_c = 0
    local cls_10 = torch.DoubleTensor(10):fill(0)
    for cls = 1, 10 do
        local cnt = 0
        for ite = 1, n_points do
            if trainset.label[ite] == cls - 1 then
                cnt = cnt + 1
            end
        end
        cls_10[cls] = cnt
        h_c = h_c - cnt/n_points*math.log(cnt/n_points)
        -- print(cnt/n_points, h_c)
    end

    for epoch = 1, n_epoches do

        sgd_params.learningRate = lr_d * math.pow(0.1, epoch / lr_decay)
        for ite = 1, 3 do 
            _, fs = optim.sgd(feval_ddcrp, x, sgd_params, state_ddcrp)
            if ite % 1 == 0 then
                print(string.format("   [%s], optimizing DDCRP, ite: %d, loss: %f", os.date("%c", os.time()), ite, fs[1]))
            end
        end

        sgd_params.learningRate = lr_a * math.pow(0.1, epoch / lr_decay)
        for ite = 1, 1000 do 
            _, fs = optim.sgd(feval_ae, x, sgd_params, state_ae)
            if ite % 100 ==  0 then
                print(string.format("   [%s], optimizing AE, ite: %d, loss: %f", os.date("%c", os.time()), ite, fs[1]))
            end
        end

        if epoch % 10 == 0 then
            -- visualize(trainset.label, model.modules[z_index].output:double(), epoch)
            torch.save(model_name .. string.format("_%d", epoch), model)
        end

        local h_o = 0
        local MI = 0
        local tp_fp = 0
        local tp = 0
        local fn_tn = 0
        local fn = 0
        local clu_num = {}
        local clu_cnt_10 = {}
        for table_id, table_i in pairs(tables) do
            if #table_i > 0 then
                h_o = h_o - (#table_i)/n_points*math.log((#table_i)/n_points)

                tp_fp = tp_fp + #table_i*(#table_i-1)/2
                for ite = 1, #clu_num do
                    fn_tn = fn_tn + clu_num[ite] * (#table_i)
                end
                table.insert(clu_num, #table_i)

                local cnt_10 = torch.DoubleTensor(10):fill(0)
                for porint_id, point in pairs(table_i) do
                    cnt_10[trainset.label[point]+1] = cnt_10[trainset.label[point]+1] + 1
                end
                for cls = 1, 10 do
                    if cnt_10[cls] > 0 then
                        MI = MI + cnt_10[cls]/n_points*math.log(n_points*cnt_10[cls]/(#table_i)/cls_10[cls])
                    end
                    tp = tp + cnt_10[cls]*(cnt_10[cls]-1)/2

                    for ite = 1, #clu_cnt_10 do
                        fn = fn + clu_cnt_10[ite][cls] * cnt_10[cls]
                    end
                end
                table.insert(clu_cnt_10, cnt_10)
            end
        end

        local tn = fn_tn - fn
        local RI = (tp+tn) / (tp_fp+fn_tn)

        print (string.format('[%s], jointly training, epoch: %d, current loss: %4f, MI: %4f, NMI: %4f, RI: %4f', os.date("%c", os.time()), epoch, fs[1], MI, MI*2/(h_o+h_c), RI))
    end
end

print(string.format("[%s], start training...", os.date("%c", os.time())))

-- initialize
n_points = 1000
n_epoches = 300
z_index = 9

local trainset = {
    size = n_points, --70000,
    data = fullset.data[{{1,n_points}}]:double()/256,--torch.cat(fullset.data[{{1,60000}}], testset.data[{{1,10000}}], 1):double() / 256,
    label = fullset.label[{{1,n_points}}]--torch.cat(fullset.label[{{1,60000}}], testset.label[{{1,10000}}],1)
}
trainset.data = trainset.data:cuda()

local inference_net = nn.Sequential():cuda()
inference_net:add(model.modules[2])
inference_net:add(model.modules[3])
inference_net:add(model.modules[4])
inference_net:add(model.modules[5])
inference_net:add(model.modules[6])
inference_net:add(model.modules[7])
inference_net:add(model.modules[8])
inference_net:add(model.modules[9])

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
inference_net:forward(trainset.data)
local z = inference_net.output:double()
for ite = 1, 200 do
    gibbs_sample(z, n_points)
end
torch.save(string.format('params_%d.t7', tag), {belong, connect, tables, connected, n_tables})

-- params = torch.load('params.t7')
-- belong = params[1] 
-- connect = params[2] 
-- tables = params[3]  
-- connected = params[4]
-- n_tables = params[5] 


train(model, inference_net, trainset, z_index, n_points, n_epoches, 7, 3, 1, 0.0001, 0.01, 50)
-- output 0.0000001 0.1 20 yes
-- output1 0.0000005 0.1 20 no
-- 0.0000001 0.1 50 0.9728 alpha=6
-- 0.0000001 0.1 50 0.988113 alpha = 5
-- 3 3 0.1 50 output
-- 3 3 0.1 100 output1
-- 3 1 0.1 50 output
-- 4 2 0.1 50 output1
-- 50 is better than 100 maybe
-- 8 2
-- 16 4
-- 1 0.0001 output_1
-- 0.3 0.0001 output
-- 0.1 0.0001 output0
-- 0.03 0.001 output1
-- 0.01 0.001 output2
-- 0.003 0.001 output3
-- 0.001 0.01 output4
-- 0.0003 0.01 output5
-- 0.0001 0.01 output6
-- 1 0.0001 25 5 output_2
-- 0.1 0.0001 50 output_3
-- 0.1 0.0003 50 output_4 12 12 12
-- 0.1 0.001 50 output_5
-- 0.1 0.003 50 output_6
-- 0.1 0.01 50 output_7

-- best 6   1   0.0001  50
-- 3 1 0.0001 0.00005 0.00001