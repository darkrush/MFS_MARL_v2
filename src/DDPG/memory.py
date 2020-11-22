import torch

class Memory(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit
        self._next_entry = 0
        self._nb_entries = 0
        self.use_cuda = False
        
        self.data_buffer = {}
        self.data_buffer['obs0'      ] = [torch.zeros((limit,) + shape) for shape in observation_shape]
        self.data_buffer['obs1'      ] = [torch.zeros((limit,) + shape) for shape in observation_shape]
        self.data_buffer['actions'   ] = torch.zeros((limit,) + action_shape)
        self.data_buffer['rewards'   ] = torch.zeros((limit,1))
        self.data_buffer['terminals1'] = torch.zeros((limit,1))

    def cuda(self):
        self.data_buffer['obs0'] = [buffer.cuda() for buffer in self.data_buffer['obs0']]
        self.data_buffer['obs1'] = [buffer.cuda() for buffer in self.data_buffer['obs1']]
        self.data_buffer['actions'] = self.data_buffer['actions'].cuda()
        self.data_buffer['rewards'] = self.data_buffer['rewards'].cuda()
        self.data_buffer['terminals1'] = self.data_buffer['terminals1'].cuda()
        self.use_cuda = True


    def __getitem(self, idx):
        return_dict = {}
        return_dict['obs0']=[tensor[idx] for tensor in self.data_buffer['obs0']]
        return_dict['obs1']=[tensor[idx] for tensor in self.data_buffer['obs1']]
        return_dict['actions']=self.data_buffer['actions'][idx]
        return_dict['rewards']=self.data_buffer['rewards'][idx]
        return_dict['terminals1']=self.data_buffer['terminals1'][idx]
        return return_dict
    
    def sample_last(self, batch_size):
        batch_idxs = torch.arange(self._next_entry - batch_size ,self._next_entry)%self._nb_entries
        return_dict = {}
        return_dict['obs0']=[torch.index_select(tensor,0,batch_idxs) for tensor in self.data_buffer['obs0']]
        return_dict['obs1']=[torch.index_select(tensor,0,batch_idxs) for tensor in self.data_buffer['obs1']]
        return_dict['actions']=torch.index_select(self.data_buffer['actions'],0,batch_idxs)
        return_dict['rewards']=torch.index_select(self.data_buffer['rewards'],0,batch_idxs)
        return_dict['terminals1']=torch.index_select(self.data_buffer['terminals1'],0,batch_idxs)
        return return_dict
    
    def sample(self, batch_size):
        batch_idxs = torch.randint(0,self._nb_entries, (batch_size,),dtype = torch.long)
        if self.use_cuda:
            batch_idxs = batch_idxs.cuda()
        return_dict = {}
        return_dict['obs0']=[torch.index_select(tensor,0,batch_idxs) for tensor in self.data_buffer['obs0']]
        return_dict['obs1']=[torch.index_select(tensor,0,batch_idxs) for tensor in self.data_buffer['obs1']]
        return_dict['actions']=torch.index_select(self.data_buffer['actions'],0,batch_idxs)
        return_dict['rewards']=torch.index_select(self.data_buffer['rewards'],0,batch_idxs)
        return_dict['terminals1']=torch.index_select(self.data_buffer['terminals1'],0,batch_idxs)
        return return_dict
    
    @property
    def nb_entries(self):
        return self._nb_entries
    
    def reset(self):
        self._next_entry = 0
        self._nb_entries = 0
        
    def append(self, obs0, action, reward, obs1, terminal1):
        for item_idx in range(len(obs0)):
            self.data_buffer['obs0'][item_idx][self._next_entry] = torch.as_tensor(obs0[item_idx])
            self.data_buffer['obs1'][item_idx][self._next_entry] = torch.as_tensor(obs1[item_idx])
        self.data_buffer['actions'][self._next_entry] = torch.as_tensor(action)
        self.data_buffer['rewards'][self._next_entry] = torch.as_tensor(reward)
        self.data_buffer['terminals1'][self._next_entry] = torch.as_tensor(terminal1)
        
        if self._nb_entries < self.limit:
            self._nb_entries += 1
            
        self._next_entry = (self._next_entry + 1)%self.limit