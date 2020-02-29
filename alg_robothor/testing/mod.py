


#percepetion
class AtariNet(nn.Module):
    def __init__(self, n_frames,
                 n_map_channels=0,
                 use_target=True,
                 output_size=512):
        super(AtariNet, self).__init__()
        self.n_frames = n_frames
        self.use_target = use_target
        self.use_map = n_map_channels > 0
        self.map_channels = n_map_channels
        self.output_size = output_size
        
        if self.use_map:
            self.map_tower = atari_conv(num_inputs=self.n_frames * self.map_channels)
        else:
            self.map_channels = 0
        if self.use_target:
            self.target_channels = 3
        else:
            self.target_channels = 0

        self.image_tower = atari_small_conv(num_inputs=self.n_frames*3)
        self.conv1 = nn.Conv2d(64 + (self.n_frames * self.target_channels), 32, 3, stride=1)
        self.flatten = Flatten()
        self.fc1 = init_(nn.Linear(32 * 7 * 7 * (self.use_map + 1), 1024))
        self.fc2 = init_(nn.Linear(1024, self.output_size))

    def forward(self, x):
        x_rgb = x['rgb_filled']
        x_rgb = self.image_tower(x_rgb)
        if self.use_target:
            x_rgb = torch.cat([x_rgb, x["target"]], dim=1)
        x_rgb = F.relu(self.conv1(x_rgb))
        if self.use_map:
            x_map = x['map']
            x_map = self.map_tower(x_map)
            x_rgb = torch.cat([x_rgb, x_map], dim=1)
        x = self.flatten(x_rgb)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


#base
class NaivelyRecurrentACModule(ActorCriticModule):
    ''' consists of a perception unit, a recurrent unit. 
        The perception unit produces a state representation P of shape internal_state_shape 
        The recurrent unit learns a function f(P) to generate a new internal_state
        The action and value should both be linear combinations of the internal state
    '''
    def __init__(self, perception_unit, use_gru=False, internal_state_size=512):
        super(NaivelyRecurrentACModule, self).__init__()
        self._internal_state_size = internal_state_size
        
        if use_gru:
            self.gru = nn.GRUCell(input_size=internal_state_size, hidden_size=internal_state_size)
            # nn.init.orthogonal_(self.gru.weight_ih.data)
            # nn.init.orthogonal_(self.gru.weight_hh.data)
            # self.gru.bias_ih.data.fill_(0)
            # self.gru.bias_hh.data.fill_(0)
        
        self.perception_unit = perception_unit
    
        # Make the critic
        init_ = lambda m: init(m,
          nn.init.orthogonal_,
          lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(internal_state_size, 1))

        #self.train()

#ac
class PolicyWithBase(BasePolicy):
    def __init__(self, base, action_space, decoder_path=None, num_stack=4, takeover:TakeoverPolicy = None, validator:ActionValidator = None):
        '''
            Args:
                base: A unit which of type ActorCriticModule
        '''
        super().__init__()
        self.base = base
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.internal_state_size = self.base.internal_state_size
        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()

        self.takeover = takeover
        self.action_validator = validator

        if decoder_path is not None:
            task = [t for t in SINGLE_IMAGE_TASKS if t in decoder_path][0]
            decoder = TaskonomyDecoder(out_channels=TASKS_TO_CHANNELS[task], eval_only=True)
            checkpoint = torch.load(decoder_path)
            decoder.load_state_dict(checkpoint['state_dict'])
            self.decoder = FrameStacked(decoder, num_stack)
            self.decoder.cuda()
            self.decoder.eval()



    def forward(self, inputs, states, masks):
        ''' TODO: find out what these do '''
        raise NotImplementedError


    def act(self, observations, model_states, masks, deterministic=False):
        ''' TODO: find out what these do 
            
            inputs: Observations?
            states: Model state?
            masks: ???
            deterministic: Boolean, True if the policy is deterministic
        '''

        value, actor_features, states = self.base(observations, model_states, masks)
        dist = self.dist(actor_features)

        if deterministic:
            # Select MAP/MLE estimate (depending on if we are feeling Bayesian)
            action = dist.mode()
        else:
            # Sample from trained posterior distribution
            action = dist.sample()

        self.probs = dist.probs
        action_log_probs = dist.log_probs(action)
        self.entropy = dist.entropy().mean()
        self.perplexity = torch.exp(self.entropy)

        # used to force agent to one action in training env (not in all envs!) useful for debugging
        # from habitat.sims.habitat_simulator import SimulatorActions
        # action[0][0] = SimulatorActions.FORWARD.value

        # apply takeover
        if self.takeover is not None:
            takeover_action, takeover_mask = self.takeover.act_with_mask(observations, model_states, masks, deterministic)
            takeover_action, takeover_mask = takeover_action.cuda(), takeover_mask.cuda()
            value.squeeze(1).masked_fill_(takeover_mask, 0).unsqueeze(1)
            action.squeeze(1).masked_scatter_(takeover_mask, takeover_action).unsqueeze(1)
            action_log_probs.squeeze(1).masked_fill_(takeover_mask, 0).unsqueeze(1)
            # states.masked_scatter_(takeover_mask, torch.zeros_like(states, device='cuda'))  TODO

        if self.action_validator is not None:
            action = self.action_validator.check_action(action)

        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks):
        ''' TODO: find out what these do '''
        value, _, _ = self.base(inputs, states, masks)
        return value

    def evaluate_actions(self, inputs, states, masks, action, cache={}):
        ''' TODO: find out what these do '''
        value, actor_features, states = self.base(inputs, states, masks, cache)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states

    def compute_intrinsic_losses(self, intrinsic_losses, inputs, states, masks, action, cache):
        losses = {}
        for intrinsic_loss in intrinsic_losses:
            if intrinsic_loss == 'activation_l2':
                assert 'residual' in cache,  f'cache does not contain residual. it contains {cache.keys()}' # (8*4) x 16 x 16
                diff = self.l2(inputs['taskonomy'], cache['residual'])
                losses[intrinsic_loss] = diff
            if intrinsic_loss == 'activation_l1':
                assert 'residual' in cache,  f'cache does not contain residual. it contains {cache.keys()}'
                diff = self.l1(inputs['taskonomy'], cache['residual'])
                losses[intrinsic_loss] = diff
            if intrinsic_loss == 'perceptual_l1':  # only L1 since decoder
                assert 'residual' in cache,  f'cache does not contain residual. it contains {cache.keys()}'
                act_teacher = self.decoder(inputs['taskonomy'])
                act_student = self.decoder(cache['residual'])     # this uses a lot of memory... make sure that ppo_num_epoch=16
                diff = self.l1(act_teacher, act_student)
                losses[intrinsic_loss] = diff
            if intrinsic_loss == 'weight':
                pass
        return losses