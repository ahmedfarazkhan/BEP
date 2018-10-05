import pickle
import numpy as np
import os
import theano
import theano.tensor as T

def rho(s):
    return T.clip(s,0.,1.)
    #return T.nnet.sigmoid(4.*s-2.)

def load_pickled_data(f):
    u = pickle._Unpickler( f )
    u.encoding = 'latin1'
    return u.load()

def load_dataset_simplified(in_path="Data/[SIM].data"):
  with open(in_path, 'rb') as ff:
    archetypes, (train_x_values, train_y_values, train_x_centroid), (test_x_values, test_y_values, test_x_centroid) = load_pickled_data(ff)#pickle.load(ff)

#  dataset = {'x_train' : np.asarray(train_images),
#             'y_train' : np.asarray(train_labels),
#             'x_test'  : np.asarray(test_images),
#             'y_test'  : np.asarray(test_labels)}
  # CONCATENATE TRAINING, VALIDATION AND TEST SETS
  x_values = list(train_x_values) + list(test_x_values)
  y_values = list(train_y_values) + list(test_y_values)
  x_centr_values = list(train_x_centroid) + list(test_x_centroid)

  x =        theano.shared(np.asarray(x_values, dtype=theano.config.floatX), borrow=True)
  y = T.cast(theano.shared(np.asarray(y_values, dtype=theano.config.floatX), borrow=True), 'int32')
  x_centr = theano.shared(np.asarray(x_centr_values, dtype=theano.config.floatX), borrow=True)
  archs = theano.shared(np.asarray(archetypes, dtype=theano.config.floatX), borrow=True)
  len_dataset = len(x_values)

  return x, y, x_centr, archs, len_dataset


# Modified for simplified datset
D = 4

class Network(object):

    def __init__(self, name, hyperparameters=dict()):

        self.path = os.getcwd() + "/Data/" + name + ".save"

        # LOAD/INITIALIZE PARAMETERS
        self.biases_fw, self.biases_bw, self.weights_fw, self.weights_bw, self.hyperparameters, self.training_curves = self.__load_params(hyperparameters)

        # LOAD EXTERNAL WORLD (=DATA)
        self.images, self.targets,  self.centroids, self.archetypes, dataset_size = load_dataset_simplified()
        
        # INITIALIZE PERSISTENT PARTICLES
        layer_sizes = [D*D] + self.hyperparameters["hidden_sizes"] + [D*D]
        values_fw = [np.zeros((dataset_size, layer_size), dtype=theano.config.floatX) for layer_size in layer_sizes[1:]]
        values_bw = [np.zeros((dataset_size, layer_size), dtype=theano.config.floatX) for layer_size in list(reversed(layer_sizes))[1:]]

        # FW: NON-INPUT LAYERS
        self.persistent_particles_fw  = [theano.shared(value, borrow=True) for value in values_fw]
        # BW: NON-OUTPUT LAYERS
        self.persistent_particles_bw  = [theano.shared(value, borrow=True) for value in values_bw]
        #self.persistent_particles_bw  = [theano.shared(value, borrow=True) for value in list(reversed(values_fw))[1:]] + [theano.shared(np.zeros((dataset_size, layer_sizes[0]), dtype=theano.config.floatX), borrow=True)]

        # LAYERS = MINI-BACTHES OF DATA + MINI-BACTHES OF PERSISTENT PARTICLES
        self.batch_size = self.hyperparameters["batch_size"]
        self.index = theano.shared(np.int32(0), name='index') # index of a mini-batch

        self.x_data = self.images[self.index * self.batch_size: (self.index + 1) * self.batch_size]
        self.y_data = self.targets[self.index * self.batch_size: (self.index + 1) * self.batch_size]
        self.y_data_one_hot = T.extra_ops.to_one_hot(self.y_data, D*D)
        self.cent_data = self.centroids[self.index * self.batch_size: (self.index + 1) * self.batch_size]

        # Todo: make inputs tensor variables so they can be modified
        self.layers_fw = [self.x_data]+[particle[self.index * self.batch_size: (self.index + 1) * self.batch_size] for particle in self.persistent_particles_fw]
        self.layers_bw = [self.y_data_one_hot]+[particle[self.index * self.batch_size: (self.index + 1) * self.batch_size] for particle in self.persistent_particles_bw]

        #print("Backward layers sizes")
        #print([layer.shape.eval() for layer in self.layers_bw])
        #print("Persistent particles")
        #print([layer.shape.eval() for layer in self.persistent_particles_fw])
        #print([layer.shape.eval() for layer in self.persistent_particles_bw])
        

        # BUILD THEANO FUNCTIONS
        self.change_mini_batch_index = self.__build_change_mini_batch_index()
        self.measure                 = self.__build_measure()
		
		
	    # SELECT CASE
        # BEP TIED
#        self.free_phase_fw              = self.__build_free_phase_fw_tied()
#        self.weakly_clamped_phase_fw    = self.__build_weakly_clamped_phase_fw_tied()
#        self.free_phase_bw              = self.__build_free_phase_bw_tied()
#        self.weakly_clamped_phase_bw    = self.__build_weakly_clamped_phase_bw_tied()
        # BEP SPLIT
        # self.free_phase_fw              = self.__build_free_phase_fw()
        # self.weakly_clamped_phase_fw    = self.__build_weakly_clamped_phase_fw()
        # self.free_phase_bw              = self.__build_free_phase_bw()
        # self.weakly_clamped_phase_bw    = self.__build_weakly_clamped_phase_bw()

        # # RBEP  SPLIT
        self.free_phase_fw              = self.__build_free_phase_fw()
        self.weakly_clamped_phase_fw    = self.__build_weakly_clamped_phase_fw_rbep()
        self.free_phase_bw              = self.__build_free_phase_bw()
        self.weakly_clamped_phase_bw    = self.__build_weakly_clamped_phase_bw_rbep()

        # RBEP TIED
#        self.free_phase_fw              = self.__build_free_phase_fw_rbep_tied()
#        self.weakly_clamped_phase_fw    = self.__build_weakly_clamped_phase_fw_rbep_tied()
#        self.free_phase_bw              = self.__build_free_phase_bw_rbep_tied()
#        self.weakly_clamped_phase_bw    = self.__build_weakly_clamped_phase_bw_rbep_tied()

    def save_params(self, epoch):
        f = open(self.path.replace('.save', '-%d_epochs.save' %(epoch)), 'wb')
        biases_values_fw  = [b.get_value() for b in self.biases_fw]
        biases_values_bw  = [b.get_value() for b in self.biases_bw]
        weights_values_fw = [W.get_value() for W in self.weights_fw]
        weights_values_bw = [W.get_value() for W in self.weights_bw]
        to_dump        = biases_values_fw, biases_values_bw, weights_values_fw, weights_values_bw, self.hyperparameters, self.training_curves   
        pickle.dump(to_dump, f, protocol=2) # For Python 2 compatability
        f.close()

    def __load_params(self, hyperparameters):

        hyper = hyperparameters

        # Glorot/Bengio weight initialization
        def initialize_layer(n_in, n_out):
            rng = np.random.RandomState()
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            return W_values
        
        if os.path.isfile(self.path):
            print("Network exists")
            f = open(self.path, 'rb')
            biases_values, weights_values, hyperparameters, training_curves = load_pickled_data(f) #pickle.load(f)
            f.close()
            for k,v in hyper.items():
                hyperparameters[k]=v
        else:
            layer_sizes = [D*D] + hyperparameters["hidden_sizes"] + [D*D]
            biases_values_fw  = [np.zeros((size,), dtype=theano.config.floatX) for size in layer_sizes]
            biases_values_bw  = [np.zeros((size,), dtype=theano.config.floatX) for size in list(reversed(layer_sizes))]
            weights_values_fw = [initialize_layer(size_pre,size_post) for size_pre,size_post in zip(layer_sizes[:-1],layer_sizes[1:])]
            weights_values_bw = [initialize_layer(size_pre,size_post) for size_pre,size_post in zip(list(reversed(layer_sizes))[:-1],list(reversed(layer_sizes))[1:])]
            training_curves = dict()
            training_curves["fw training error"]   = list()
            training_curves["fw validation error"] = list()
            training_curves["bw training error"]   = list()
            training_curves["bw validation error"] = list()

        biases_fw  = [theano.shared(value=value, borrow=True) for value in biases_values_fw]
        biases_bw  = [theano.shared(value=value, borrow=True) for value in biases_values_bw]
        weights_fw = [theano.shared(value=value, borrow=True) for value in weights_values_fw]
        weights_bw = [theano.shared(value=value, borrow=True) for value in weights_values_bw]

        return biases_fw, biases_bw, weights_fw, weights_bw, hyperparameters, training_curves

    # SET INDEX OF THE MINI BATCH
    def __build_change_mini_batch_index(self):

        index_new = T.iscalar("index_new")

        change_mini_batch_index = theano.function(
            inputs=[index_new],
            outputs=[],
            updates=[(self.index,index_new)]
        )

        return change_mini_batch_index


    # ENERGY FUNCTION, DENOTED BY E
    def __energy(self, layers, weights_fw, weights_bw, biases):
        layers_bw = list(reversed(layers))
        squared_norm    =   sum( [T.batched_dot(rho(layer),rho(layer))       for layer      in layers] ) / 2.
        linear_terms    = - sum( [T.dot(rho(layer),b)                        for layer,b    in zip(layers,biases)] )
        #      ([batch, pre] . [pre, post]) <> [batch, post]                                       
        quadratic_terms_fw = - sum( [T.batched_dot(T.dot(rho(pre),W),rho(post)) for pre,W,post in zip(layers[:-1],weights_fw,layers[1:])] )
        quadratic_terms_bw = - sum( [T.batched_dot(T.dot(rho(pre),W),rho(post)) for pre,W,post in zip(layers_bw[:-1],weights_bw,layers_bw[1:])] )
        
        # Average quadratic Hopfield terms
        return squared_norm + linear_terms + ((quadratic_terms_fw + quadratic_terms_bw)/2.0)

    # COST FUNCTION, DENOTED BY C
    def __cost(self, layer, direction): # FW but works for BW with one hot x-pred value for layer
        if direction == "fw":
            return ((layer - self.y_data_one_hot) ** 2).sum(axis=1)
        elif direction == "bw":
            # CLOSEST ARCHETYPE IS THE SAME AS THE Y-LABEL
            #arches_reshaped = T.reshape(self.archetypes, [1, D*D, D*D])
            #output_reshaped = T.reshape(layer, [self.batch_size, 1, D*D])

            #sum_squared = ((arches_reshaped - output_reshaped) ** 2).sum(axis=2)

            #closest_arch = T.argmin(sum_squared, axis=1)  
            #closest_arch_one_hot = T.extra_ops.to_one_hot(closest_arch, D*D)
            
            #return ((closest_arch_one_hot- self.y_data_one_hot) ** 2).sum(axis=1)
            
            return ((layer - self.cent_data) ** 2).sum(axis=1)

    # TOTAL ENERGY FUNCTION, DENOTED BY F
    def __total_energy(self, layers, weights_fw, weights_bw, biases, clamping_factor, direction):
        return self.__energy(layers, weights_fw, weights_bw, biases) + clamping_factor * self.__cost(layers[-1], direction)
    
    # MEASURES THE ENERGY, THE COST AND THE MISCLASSIFICATION ERROR FOR THE CURRENT STATE OF THE NETWORK
    def __build_measure(self): 
       
        #print(self.archetypes.shape.eval())
        #print(self.layers_bw[-1].shape.eval())

        ### FW
        E_fw = T.mean(self.__energy(self.layers_fw, self.weights_fw, self.weights_bw, self.biases_fw))
        C_fw = T.mean(self.__cost(self.layers_fw[-1], direction="fw"))
        y_prediction = T.argmax(self.layers_fw[-1], axis=1)
        # Error count for y-error
        error_fw        = T.mean(T.neq(y_prediction, self.y_data))

        ### BW
        #print([layer.shape.eval() for layer in self.layers_fw])
        E_bw = T.mean(self.__energy(self.layers_bw, self.weights_bw, self.weights_fw, self.biases_bw))
        # IDENTIFY CLOSEST ARCHETYPE W/ MINIMAL SUM-SQUARED DISTANCE        
        # CLOSEST ARCHETYPE IS THE SAME AS THE Y-LABEL
        arches_reshaped = T.reshape(self.archetypes, [1, D*D, D*D])
        output_reshaped = T.reshape(self.layers_bw[-1], [self.batch_size, 1, D*D])

        sum_squared = ((arches_reshaped - output_reshaped) ** 2).sum(axis=2)

        closest_arch = T.argmin(sum_squared, axis=1)  
        #closest_arch_one_hot = T.extra_ops.to_one_hot(closest_arch, D*D)
        #C_bw = T.mean(self.__cost(closest_arch_one_hot)) 
        
        C_bw = T.mean(self.__cost(self.layers_bw[-1], direction="bw"))
        
        # Index of closest archetype for x-error
        error_bw        = T.mean(T.neq(closest_arch, self.y_data))

        measure = theano.function(
            inputs=[],
            outputs=[E_fw, C_fw, error_fw, E_bw, C_bw, error_bw]#, closest_arch, self.y_data]
        )

        return measure

    def __build_free_phase_fw(self):

        n_iterations = T.iscalar('n_iterations')
        epsilon  = T.fscalar('epsilon')

        def step(*layers):
            E_sum = T.sum(self.__energy(layers, self.weights_fw, self.weights_bw, self.biases_fw))
            layers_dot = T.grad(-E_sum, list(layers)) # temporal derivative of the state (free trajectory)
            layers_new = [layers[0]]+[T.clip(layer+epsilon*dot,0.,1.) for layer,dot in zip(layers,layers_dot)][1:]
            return layers_new

        ( layers, updates ) = theano.scan(
            step,
            outputs_info=self.layers_fw,
            n_steps=n_iterations
        )
        layers_end = [layer[-1] for layer in layers]

        for particles,layer,layer_end in zip(self.persistent_particles_fw,self.layers_fw[1:],layers_end[1:]):
            updates[particles] = T.set_subtensor(layer,layer_end)
        
        free_phase = theano.function(
            inputs=[n_iterations,epsilon],
            outputs=[],
            updates=updates
        )

        return free_phase
        
    def __build_free_phase_fw_tied(self):

        n_iterations = T.iscalar('n_iterations')
        epsilon  = T.fscalar('epsilon')

        def step(*layers):
            E_sum = T.sum(self.__energy(layers, self.weights_fw, [w.T for w in self.weights_fw[::-1]], self.biases_fw))
            layers_dot = T.grad(-E_sum, list(layers)) # temporal derivative of the state (free trajectory)
            layers_new = [layers[0]]+[T.clip(layer+epsilon*dot,0.,1.) for layer,dot in zip(layers,layers_dot)][1:]
            return layers_new

        ( layers, updates ) = theano.scan(
            step,
            outputs_info=self.layers_fw,
            n_steps=n_iterations
        )
        layers_end = [layer[-1] for layer in layers]

        for particles,layer,layer_end in zip(self.persistent_particles_fw,self.layers_fw[1:],layers_end[1:]):
            updates[particles] = T.set_subtensor(layer,layer_end)
        
        free_phase = theano.function(
            inputs=[n_iterations,epsilon],
            outputs=[],
            updates=updates
        )

        return free_phase

    def __build_free_phase_bw(self):

        n_iterations = T.iscalar('n_iterations')
        epsilon  = T.fscalar('epsilon')

        def step(*layers):
            E_sum = T.sum(self.__energy(layers, self.weights_bw, self.weights_fw, self.biases_bw ))
            layers_dot = T.grad(-E_sum, list(layers)) 
            layers_new = [layers[0]]+[T.clip(layer+epsilon*dot,0.,1.) for layer,dot in zip(layers,layers_dot)][1:]
            return layers_new

        ( layers, updates ) = theano.scan(
            step,
            outputs_info=self.layers_bw,
            n_steps=n_iterations
        )

        layers_end = [layer[-1] for layer in layers]

        for particles,layer,layer_end in zip(self.persistent_particles_bw,self.layers_bw[1:],layers_end[1:]):
            updates[particles] = T.set_subtensor(layer,layer_end)
        
        free_phase = theano.function(
            inputs=[n_iterations,epsilon],
            outputs=[],
            updates=updates
        )


        return free_phase
        
    def __build_free_phase_bw_tied(self):

        n_iterations = T.iscalar('n_iterations')
        epsilon  = T.fscalar('epsilon')

        def step(*layers):
            E_sum = T.sum(self.__energy(layers, self.weights_bw, [w.T for w in self.weights_bw[::-1]], self.biases_bw ))
            layers_dot = T.grad(-E_sum, list(layers)) 
            layers_new = [layers[0]]+[T.clip(layer+epsilon*dot,0.,1.) for layer,dot in zip(layers,layers_dot)][1:]
            return layers_new

        ( layers, updates ) = theano.scan(
            step,
            outputs_info=self.layers_bw,
            n_steps=n_iterations
        )

        layers_end = [layer[-1] for layer in layers]

        for particles,layer,layer_end in zip(self.persistent_particles_bw,self.layers_bw[1:],layers_end[1:]):
            updates[particles] = T.set_subtensor(layer,layer_end)
        
        free_phase = theano.function(
            inputs=[n_iterations,epsilon],
            outputs=[],
            updates=updates
        )


        return free_phase

    def __build_weakly_clamped_phase_fw(self):

        n_iterations = T.iscalar('n_iterations')
        epsilon  = T.fscalar('epsilon')
        beta = T.fscalar('beta')
        learning_rates = [T.fscalar("lr_W"+str(r+1)) for r in range(len(self.weights_fw))] # FW


        def step(*layers):
            F_sum = T.sum(self.__total_energy(layers, self.weights_fw, self.weights_bw, self.biases_fw, beta, direction="fw")) # FW energy
            layers_dot = T.grad(-F_sum, list(layers)) # temporal derivative of the state (weakly clamped trajectory)
            layers_new = [layers[0]]+[T.clip(layer+epsilon*dot,0.,1.) for layer,dot in zip(layers,layers_dot)][1:]
            return layers_new

        ( layers, updates ) = theano.scan(  
            step,
            outputs_info=self.layers_fw,
            n_steps=n_iterations
        )
        
        layers_weakly_clamped = [layer[-1] for layer in layers]

        E_mean_free           = T.mean(self.__energy(self.layers_fw, self.weights_fw, self.weights_bw, self.biases_fw))
        E_mean_weakly_clamped = T.mean(self.__energy(layers_weakly_clamped, self.weights_fw, self.weights_bw, self.biases_fw))
        biases_dot            = T.grad( (E_mean_weakly_clamped-E_mean_free) / beta, self.biases_fw,  consider_constant=layers_weakly_clamped)
        weights_dot_fw           = T.grad( (E_mean_weakly_clamped-E_mean_free) / beta, self.weights_fw, consider_constant=layers_weakly_clamped)
        weights_dot_bw           = T.grad( (E_mean_weakly_clamped-E_mean_free) / beta, self.weights_bw, consider_constant=layers_weakly_clamped)

        biases_new  = [b - lr * dot for b,lr,dot in zip(self.biases_fw[1:],learning_rates,biases_dot[1:])]
        weights_new_fw = [W - lr * dot for W,lr,dot in zip(self.weights_fw,   learning_rates,weights_dot_fw)]
        weights_new_bw = [W - lr * dot for W,lr,dot in zip(self.weights_bw,   list(reversed(learning_rates)),weights_dot_bw)]
        
        Delta_log_fw = [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights_fw,weights_new_fw)]

        #Delta_log_bw = [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights_bw,weights_new_bw)]
        
        for bias, bias_new in zip(self.biases_fw[1:],biases_new ):
            updates[bias]=bias_new
        for weight_fw, weight_new_fw in zip(self.weights_fw,weights_new_fw):
            updates[weight_fw]=weight_new_fw
        for weight_bw, weight_new_bw in zip(self.weights_bw,weights_new_bw):
            updates[weight_bw]=weight_new_bw

        weakly_clamped_phase = theano.function(
            inputs=[n_iterations, epsilon, beta]+learning_rates,
            outputs=Delta_log_fw,
            updates=updates
        )


        return weakly_clamped_phase
        

    def __build_weakly_clamped_phase_fw_rbep(self):

        n_iterations = T.iscalar('n_iterations')
        epsilon  = T.fscalar('epsilon')
        beta = T.fscalar('beta')
        learning_rates = [T.fscalar("lr_W"+str(r+1)) for r in range(len(self.weights_fw))] # FW


        def step(*layers):
            F_sum = T.sum(self.__total_energy(layers, self.weights_fw, self.weights_bw, self.biases_fw, beta, direction="fw")) # FW energy
            layers_dot = T.grad(-F_sum, list(layers)) # temporal derivative of the state (weakly clamped trajectory)
            layers_new = [layers[0]]+[T.clip(layer+epsilon*dot,0.,1.) for layer,dot in zip(layers,layers_dot)][1:]
            return layers_new

        ( layers, updates ) = theano.scan(  
            step,
            outputs_info=self.layers_fw,
            n_steps=n_iterations
        )
        
        layers_weakly_clamped = [layer[-1] for layer in layers]

        E_mean_free           = T.mean(self.__energy(self.layers_fw, self.weights_fw, self.weights_bw, self.biases_fw))
        E_mean_weakly_clamped = T.mean(self.__energy(layers_weakly_clamped, self.weights_fw, self.weights_bw, self.biases_fw))
        biases_dot            = T.grad( (E_mean_weakly_clamped-E_mean_free) / beta, self.biases_fw,  consider_constant=layers_weakly_clamped)
        weights_dot_fw           = T.grad( (E_mean_weakly_clamped-E_mean_free) / beta, self.weights_fw, consider_constant=layers_weakly_clamped)

        biases_new  = [b - lr * dot for b,lr,dot in zip(self.biases_fw[1:],learning_rates,biases_dot[1:])]
        weights_new_fw = [W - lr * dot for W,lr,dot in zip(self.weights_fw,   learning_rates,weights_dot_fw)]
        
        Delta_log_fw = [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights_fw,weights_new_fw)]

        for bias, bias_new in zip(self.biases_fw[1:],biases_new ):
            updates[bias]=bias_new
        for weight_fw, weight_new_fw in zip(self.weights_fw,weights_new_fw):
            updates[weight_fw]=weight_new_fw

        weakly_clamped_phase = theano.function(
            inputs=[n_iterations, epsilon, beta]+learning_rates,
            outputs=Delta_log_fw,
            updates=updates
        )


        return weakly_clamped_phase

        
        
    def __build_weakly_clamped_phase_fw_tied(self):

        n_iterations = T.iscalar('n_iterations')
        epsilon  = T.fscalar('epsilon')
        beta = T.fscalar('beta')
        learning_rates = [T.fscalar("lr_W"+str(r+1)) for r in range(len(self.weights_fw))] # FW

        def step(*layers):
            F_sum = T.sum(self.__total_energy(layers, self.weights_fw, [w.T for w in self.weights_fw[::-1]], self.biases_fw, beta, direction="fw")) # FW energy
            layers_dot = T.grad(-F_sum, list(layers)) # temporal derivative of the state (weakly clamped trajectory)
            layers_new = [layers[0]]+[T.clip(layer+epsilon*dot,0.,1.) for layer,dot in zip(layers,layers_dot)][1:]
            return layers_new

        ( layers, updates ) = theano.scan(  
            step,
            outputs_info=self.layers_fw,
            n_steps=n_iterations
        )
        
        layers_weakly_clamped = [layer[-1] for layer in layers]

        E_mean_free           = T.mean(self.__energy(self.layers_fw, self.weights_fw, [w.T for w in self.weights_fw[::-1]], self.biases_fw))
        E_mean_weakly_clamped = T.mean(self.__energy(layers_weakly_clamped, self.weights_fw, [w.T for w in self.weights_fw[::-1]], self.biases_fw))
        biases_dot            = T.grad( (E_mean_weakly_clamped-E_mean_free) / beta, self.biases_fw,  consider_constant=layers_weakly_clamped)
        weights_dot_fw           = T.grad( (E_mean_weakly_clamped-E_mean_free) / beta, self.weights_fw, consider_constant=layers_weakly_clamped)
        #weights_dot_bw           = T.grad( (E_mean_weakly_clamped-E_mean_free) / beta, self.weights_bw, consider_constant=layers_weakly_clamped)

        biases_new  = [b - lr * dot for b,lr,dot in zip(self.biases_fw[1:],learning_rates,biases_dot[1:])]
        weights_new_fw = [W - lr * dot for W,lr,dot in zip(self.weights_fw,   learning_rates,weights_dot_fw)]
   
        Delta_log_fw = [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights_fw,weights_new_fw)]

        for bias, bias_new in zip(self.biases_fw[1:],biases_new ):
            updates[bias]=bias_new
        for weight_fw, weight_new_fw in zip(self.weights_fw,weights_new_fw):
            updates[weight_fw]=weight_new_fw

        weakly_clamped_phase = theano.function(
            inputs=[n_iterations, epsilon, beta]+learning_rates,
            outputs=Delta_log_fw,
            updates=updates
        )

        return weakly_clamped_phase

        
    def __build_weakly_clamped_phase_bw(self):

        n_iterations = T.iscalar('n_iterations')
        epsilon  = T.fscalar('epsilon')
        alpha = T.fscalar('alpha')
        learning_rates = [T.fscalar("lr_W"+str(r+1)) for r in range(len(self.weights_fw))] # Actually reversed from FW case


        def step(*layers):
            F_sum = T.sum(self.__total_energy(layers, self.weights_bw, self.weights_fw, self.biases_bw, alpha, direction="bw")) # BW energy
            layers_dot = T.grad(-F_sum, list(layers)) # temporal derivative of the state (weakly clamped trajectory)
            layers_new = [layers[0]]+[T.clip(layer+epsilon*dot,0.,1.) for layer,dot in zip(layers,layers_dot)][1:]
            return layers_new

        ( layers, updates ) = theano.scan(  
            step,
            outputs_info=self.layers_bw,
            n_steps=n_iterations
        )
        
        layers_weakly_clamped = [layer[-1] for layer in layers]

        E_mean_free           = T.mean(self.__energy(self.layers_bw, self.weights_bw, self.weights_fw, self.biases_bw ))
        E_mean_weakly_clamped = T.mean(self.__energy(layers_weakly_clamped, self.weights_bw, self.weights_fw, self.biases_bw ))
        biases_dot            = T.grad( (E_mean_weakly_clamped-E_mean_free) / alpha, self.biases_bw,  consider_constant=layers_weakly_clamped)
        weights_dot_fw           = T.grad( (E_mean_weakly_clamped-E_mean_free) / alpha, self.weights_fw, consider_constant=layers_weakly_clamped)
        weights_dot_bw           = T.grad( (E_mean_weakly_clamped-E_mean_free) / alpha, self.weights_bw, consider_constant=layers_weakly_clamped)

        biases_new  = [b - lr * dot for b,lr,dot in zip(self.biases_bw[1:],learning_rates,biases_dot[1:])]
        weights_new_fw = [W - lr * dot for W,lr,dot in zip(self.weights_fw,   list(reversed(learning_rates)),weights_dot_fw)]
        weights_new_bw = [W - lr * dot for W,lr,dot in zip(self.weights_bw,   learning_rates,weights_dot_bw)]#list(reversed(learning_rates)),weights_dot_bw)]
        
        Delta_log_fw = [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights_fw,weights_new_fw)]

        Delta_log_bw = [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights_bw,weights_new_bw)]
        
                
        for bias, bias_new in zip(self.biases_bw[1:],biases_new ):
            updates[bias]=bias_new
        for weight_fw, weight_new_fw in zip(self.weights_fw,weights_new_fw):
            updates[weight_fw]=weight_new_fw
        for weight_bw, weight_new_bw in zip(self.weights_bw,weights_new_bw):
            updates[weight_bw]=weight_new_bw

        weakly_clamped_phase = theano.function(
            inputs=[n_iterations, epsilon, alpha]+learning_rates,
            outputs=Delta_log_bw,
            updates=updates
        )


        return weakly_clamped_phase

        
                
    def __build_weakly_clamped_phase_bw_rbep(self):

        n_iterations = T.iscalar('n_iterations')
        epsilon  = T.fscalar('epsilon')
        alpha = T.fscalar('alpha')
        learning_rates = [T.fscalar("lr_W"+str(r+1)) for r in range(len(self.weights_fw))] # Actually reversed from FW case


        def step(*layers):
            F_sum = T.sum(self.__total_energy(layers, self.weights_bw, self.weights_fw, self.biases_bw, alpha, direction="bw")) # BW energy
            layers_dot = T.grad(-F_sum, list(layers)) # temporal derivative of the state (weakly clamped trajectory)
            layers_new = [layers[0]]+[T.clip(layer+epsilon*dot,0.,1.) for layer,dot in zip(layers,layers_dot)][1:]
            return layers_new

        ( layers, updates ) = theano.scan(  
            step,
            outputs_info=self.layers_bw,
            n_steps=n_iterations
        )
        
        layers_weakly_clamped = [layer[-1] for layer in layers]

        E_mean_free           = T.mean(self.__energy(self.layers_bw, self.weights_bw, self.weights_fw, self.biases_bw ))
        E_mean_weakly_clamped = T.mean(self.__energy(layers_weakly_clamped, self.weights_bw, self.weights_fw, self.biases_bw ))
        biases_dot            = T.grad( (E_mean_weakly_clamped-E_mean_free) / alpha, self.biases_bw,  consider_constant=layers_weakly_clamped)
        weights_dot_bw           = T.grad( (E_mean_weakly_clamped-E_mean_free) / alpha, self.weights_bw, consider_constant=layers_weakly_clamped)

        biases_new  = [b - lr * dot for b,lr,dot in zip(self.biases_bw[1:],learning_rates,biases_dot[1:])]
        weights_new_bw = [W - lr * dot for W,lr,dot in zip(self.weights_bw,   learning_rates,weights_dot_bw)]#list(reversed(learning_rates)),weights_dot_bw)]

        Delta_log_bw = [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights_bw,weights_new_bw)]
        
                
        for bias, bias_new in zip(self.biases_bw[1:],biases_new ):
            updates[bias]=bias_new

        for weight_bw, weight_new_bw in zip(self.weights_bw,weights_new_bw):
            updates[weight_bw]=weight_new_bw

        weakly_clamped_phase = theano.function(
            inputs=[n_iterations, epsilon, alpha]+learning_rates,
            outputs=Delta_log_bw,
            updates=updates
        )


        return weakly_clamped_phase
        
    def __build_weakly_clamped_phase_bw_tied(self):

        n_iterations = T.iscalar('n_iterations')
        epsilon  = T.fscalar('epsilon')
        alpha = T.fscalar('alpha')
        learning_rates = [T.fscalar("lr_W"+str(r+1)) for r in range(len(self.weights_fw))] # Actually reversed from FW case


        def step(*layers):
            F_sum = T.sum(self.__total_energy(layers, self.weights_bw, [w.T for w in self.weights_bw[::-1]], self.biases_bw, alpha, direction="bw")) # BW energy
            layers_dot = T.grad(-F_sum, list(layers)) # temporal derivative of the state (weakly clamped trajectory)
            layers_new = [layers[0]]+[T.clip(layer+epsilon*dot,0.,1.) for layer,dot in zip(layers,layers_dot)][1:]
            return layers_new

        ( layers, updates ) = theano.scan(  
            step,
            outputs_info=self.layers_bw,
            n_steps=n_iterations
        )
        
        layers_weakly_clamped = [layer[-1] for layer in layers]

        E_mean_free           = T.mean(self.__energy(self.layers_bw, self.weights_bw, [w.T for w in self.weights_bw[::-1]], self.biases_bw ))
        E_mean_weakly_clamped = T.mean(self.__energy(layers_weakly_clamped, self.weights_bw, [w.T for w in self.weights_bw[::-1]], self.biases_bw ))
        biases_dot            = T.grad( (E_mean_weakly_clamped-E_mean_free) / alpha, self.biases_bw,  consider_constant=layers_weakly_clamped)
        weights_dot_bw           = T.grad( (E_mean_weakly_clamped-E_mean_free) / alpha, self.weights_bw, consider_constant=layers_weakly_clamped)

        biases_new  = [b - lr * dot for b,lr,dot in zip(self.biases_bw[1:],learning_rates,biases_dot[1:])]
        weights_new_bw = [W - lr * dot for W,lr,dot in zip(self.weights_bw,   learning_rates,weights_dot_bw)]

        Delta_log_bw = [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights_bw,weights_new_bw)]
        
                
        for bias, bias_new in zip(self.biases_bw[1:],biases_new ):
            updates[bias]=bias_new
        for weight_bw, weight_new_bw in zip(self.weights_bw,weights_new_bw):
            updates[weight_bw]=weight_new_bw

        weakly_clamped_phase = theano.function(
            inputs=[n_iterations, epsilon, alpha]+learning_rates,
            outputs=Delta_log_bw,
            updates=updates
        )


        return weakly_clamped_phase