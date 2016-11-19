require 'torch'
local pretty = require 'pl.pretty'
local lapp = require 'pl.lapp'
lapp.slack = true

local mnist = require 'mnist'
require 'optim'

-- some command line parameters
local opt = lapp[[
-b,--batch_size     (default 32)                Batch size
--LR                (default 1e-3)              Learning rate
--L2                (default 1e-3)              Weight decay / L2 regularization
-g,--gpu            (default 0)                 GPU idx (set zero for CPU)
--seed              (default 42)                Random seed to initialize weights
--max_epochs        (default 100)               Max. epochs
-h,--help                                       Print this message
]]

if opt.help then
    print(opt)
    print('Example usage: th mnist.lua -b 256 --LR 1e-3 -g 0 --max_epochs 25')
    os.exit()
end

-- random seeds are used to initialize weights
-- important to set seed in both main torch and cuda
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
if opt.gpu > 0 then
    print('Using GPU')
    require 'cutorch'
    require 'nn'
    require 'cunn'
    cutorch.setDevice(opt.gpu)
    cutorch.manualSeedAll(opt.seed)
else
    -- fakecuda allows dummy :cuda() calls without a GPU
    print('Using CPU')
    require 'nn'
    require('fakecuda').init(true)
end

-- torch has a package that loads mnsit
-- install it using "luarocks install mnist"
function load_mnist()
    local train, test = mnist.traindataset(), mnist.testdataset()
    train.data = train.data:reshape(60000, 1, 28, 28):float()
    test.data = test.data:reshape(10000, 1, 28, 28):float()

    -- scale data to lie between [-1,1]
    train.data:add(-126):div(126)
    -- torch likes 1-indexed labels, so we add 1 to add classes
    -- which are originally from [0,9]
    train.label:add(1)

    -- will just use the test set as validation set here
    test.data:add(-126):div(126)
    test.label:add(1)

    return {train.data, train.label}, {test.data, test.label}
end

-- build a LeNet model for MNIST
-- should converge to ~0.75% generalization error
function build_model()
    -- a block of convolution - ReLU - max-pooling - batch-normalization - dropout
    -- fin: number of input filters for the convolutional layer
    -- fout: number of output filters
    -- psz: receptive field for max-pooling
    local function block(fin, fout, psz)
        return nn.Sequential()
            :add(nn.SpatialConvolution(fin, fout, 5,5))
            :add(nn.ReLU())
            :add(nn.SpatialMaxPooling(psz,psz,psz,psz))
            :add(nn.SpatialBatchNormalization(fout,nil,nil,false))
            :add(nn.Dropout(0.5))
    end

    local m = nn:Sequential()
        :add(block(1,20,3))             -- grayscale input, 20 output channels
        :add(block(20,50,2))            -- 20 input channels, 50 output channels
        :add(nn.View(50*2*2))           -- resize the channels to look like a vector
        :add(nn.Linear(50*2*2, 256))    -- linear layer, 50*2*2 inputs, 256 outputs
        :add(nn.ReLU())
        :add(nn.Dropout(0.5))
        :add(nn.Linear(256,10))         -- final layer, 256 inputs, 10 outputs
        -- use cross entropy loss to classify
        -- LogSoftMax is log(e^{x_i}/sum_i e^{-x_i}) for each element of the vector
        -- x, note that x is of size 10
        :add(nn.LogSoftMax()) 

        -- the cross entropy loss in torch is called ClassNLLCriterion
    return m:cuda(), nn.ClassNLLCriterion():cuda()
end

function trainer(model, cost, d)
    local x, y = d[1], d[2]

    -- training mode sets the neurons to use dropout
    model:training()

    -- w are all the parameters arranged as a vector
    -- dw are all the gradients arranged as a vector
    local w, dw = model:getParameters()

    local num_batches = x:size(1)/opt.batch_size
    local bs = opt.batch_size
    local loss = 0

    -- confusion computes the confusion matrix
    confusion:zero()
    for b =1,num_batches do
        collectgarbage()

        -- randomly sample batch_size elements from the training set
        local idx = torch.Tensor(bs):random(1, x:size(1)):type('torch.LongTensor')
        local xc, yc = x:index(1, idx), y:index(1, idx):cuda()

        -- evaluates the forward and backward pass on the network
        -- and the loss function, gradients are given out as dw
        -- torch has assignment by reference, so dw and model:getParameters()[2]
        -- are always the exact same memory locations
        local feval = function(_w)
            if _w ~= w then w:copy(_w) end
            dw:zero()

            local yh = model:forward(xc)
            local f = cost:forward(yh, yc)
            local dfdy = cost:backward(yh, yc)
            model:backward(xc, dfdy)
            cutorch.synchronize()
            
            loss = loss + f
            confusion:batchAdd(yh, yc)
            confusion:updateValids()

            return f, dw
        end

        -- We use Adam to optimize, feval is required to
        -- return the loss on the current batch and the gradients
        -- computed via backpropagation for it
        optim.adam(feval, w, optim_state)

        if b % 10 == 0 then
            print( ('+[%2d][%3d/%3d] Loss: %.5f Error: %.3f %%'):format(epoch, b, num_batches, loss/b, (1 - confusion.totalValid)*100))
        end
    end
    print( ('*[%2d] Loss: %.5f Error: %.3f %%'):format(epoch, loss/num_batches, (1 - confusion.totalValid)*100))    
end

function tester(model, cost, d)
    local x, y = d[1], d[2]
    model:evaluate()

    local num_batches = x:size(1)/opt.batch_size
    local bs = opt.batch_size

    confusion:zero()
    for b =1,num_batches do
        collectgarbage()

        -- Linearly iterate over the test dataset
        local sidx,eidx = (b-1)*bs, math.min(b*bs, x:size(1))
        local xc, yc =  x:narrow(1, sidx + 1, eidx-sidx):cuda(),
                        y:narrow(1, sidx + 1, eidx-sidx):cuda()

        local yh = model:forward(xc)
        local f = cost:forward(yh, yc)
        cutorch.synchronize()
    
        confusion:batchAdd(yh, yc)
        confusion:updateValids()
        
        if b % 10 == 0 then
            print( ('++[%2d][%3d/%3d] Error: %.3f %%'):format(epoch, b, num_batches, (1 - confusion.totalValid)*100))
        end
    end
    print( ('**[%2d] Error: %.3f %%'):format(epoch, (1 - confusion.totalValid)*100))
    print(confusion)
end

function main()
    -- the names of classes are immaterial, they are simply used to display rows
    -- of the confusion matrix
    -- these three are "global" variables which we have used for convenience
    classes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    confusion = optim.ConfusionMatrix(classes)
    optim_state = { learningRate= opt.LR,
                    weightDecay = opt.L2}

    local model, cost = build_model()
    local train, test = load_mnist()

    epoch = 1
    while epoch <= opt.max_epochs do
        --trainer(model, cost, train)
        tester(model, cost, test)
        epoch = epoch + 1
        --torch.save('lenet.t7', model)
    end
end

main()