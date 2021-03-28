import torch
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
import types
from torch.autograd import Variable
import numpy


def init_control_gates(m):
    global relu_counter
    name = m.__class__.__name__
    if name.find('ReLU') != -1:
        m.control_gates = Parameter(torch.FloatTensor(relu_shapes[relu_counter].shape))
        m.control_gates.data = torch.from_numpy(numpy.random.uniform(0, 1, m.control_gates.data.shape)).float()
        relu_counter += 1
        relu_counter = relu_counter % len(relu_shapes)

def reset_control_gates(m):
    name = m.__class__.__name__
    if name.find('ReLU') != -1:
        m.control_gates.data.fill_(1.0)
        m.control_gates.grad.data.fill_(0.0)

def new_forward(self, x):
    out = F.relu(x, self.inplace)
    out = self.control_gates * out
    return out

def replace(m):
    name = m.__class__.__name__
    if name.find('ReLU') != -1:
        m.forward = types.MethodType(new_forward, m)

def collect_control_gates(m):
    name = m.__class__.__name__
    if name.find('ReLU') != -1:
        control_gates.append(m.control_gates)


control_gates = []
relu_shapes = []
relu_counter = 0


def cdrp_sparsity(cgs):
    deads = 0
    n = 0
    for i, cg in enumerate(cgs):
        cg_copy = torch.zeros(cg.shape)
        cg_copy[cg <= 0] = 1
        deads += torch.sum(cg_copy.flatten())
        n += len(cg_copy.flatten())
    return 100*deads/n


def get_path(model, data, target, percentile, shapes, lambd=0.01):
    global control_gates, relu_shapes, relu_counter

    control_gates = []
    relu_counter = 0
    relu_shapes = shapes

    data_var = Variable(data).cuda()
    target_var = Variable(torch.tensor(target)).cuda()
    self_predicted_output = model(data_var)
    self_pred = self_predicted_output.data.max(1)[1]
    self_predicted_prob = F.softmax(self_predicted_output)
    self_predicted_prob_var = Variable(self_predicted_prob.data.detach().clone())

    model.apply(init_control_gates)
    model.apply(replace)
    model.apply(collect_control_gates)
    model.cuda()
    
    optimizer = optim.SGD(control_gates, lr=0.01, momentum=0.9, weight_decay=0)
    
    min_loss = 1e10

    cg_list = None
    c = 0
    lambd *= 10
    while cg_list is None and c < 100:
      c += 1
      lambd /= 10
      for i in range(1000):
        output = model(data_var)
        prob = F.softmax(output)

        pred = output.data.max(1)[1]

        loss = - (self_predicted_prob_var * torch.log(prob + 1e-20)).sum(1)

        for v in control_gates:
            loss += lambd * v.abs().sum()

        if pred[0] == self_pred[0]:
            if loss.data[0] < min_loss:
                cg_list = []
                for v in control_gates:
                    cg_list.append(v.data.clone())

                min_loss = loss.data[0]
                best_output = output.data.clone()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for v in control_gates:
            v.data.clamp_(0, 100)

    model.apply(reset_control_gates)
    return cg_list


