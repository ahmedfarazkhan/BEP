from model import Network
import numpy as np
from sys import stdout, maxsize
import time
import pickle 

seed = np.random.randint(1000)
rng = np.random.seed(seed)
print("Seed was:", seed)

def train_net(net):

    path         = net.path
    hidden_sizes = net.hyperparameters["hidden_sizes"]
    n_epochs     = net.hyperparameters["n_epochs"]
    batch_size   = net.hyperparameters["batch_size"]
    n_it_neg     = net.hyperparameters["n_it_neg"]
    n_it_pos     = net.hyperparameters["n_it_pos"]
    n_it_neg_bw     = net.hyperparameters["n_it_neg_bw"]
    n_it_pos_bw     = net.hyperparameters["n_it_pos_bw"]
    epsilon      = net.hyperparameters["epsilon"]
    alpha         = net.hyperparameters["alpha"]
    beta         = net.hyperparameters["beta"]
    learning_rates  = net.hyperparameters["learning_rates"]

    print("name = %s" % (path))
    print("architecture = 16-"+"-".join([str(n) for n in hidden_sizes])+"-16")
    print("number of epochs = %i" % (n_epochs))
    print("batch_size = %i" % (batch_size))
    print("n_it_neg = %i"   % (n_it_neg))
    print("n_it_pos = %i"   % (n_it_pos))
    print("epsilon = %.1f" % (epsilon))
    print("alpha = %.1f" % (alpha))
    print("beta = %.1f" % (beta))
    print("learning rates: "+" ".join(["lr_W%i=%.3f" % (k+1,lr) for k,lr in enumerate(learning_rates)])+"\n")

    n_batches_train = 12000 / batch_size #50000 / batch_size
    n_batches_valid = 4000 / batch_size #10000 / batch_size

    start_time = time.clock()

    for epoch in range(n_epochs):

        ### TRAINING ###

        # CUMULATIVE SUM OF TRAINING ENERGY, TRAINING COST AND TRAINING ERROR
        measures_sum = [0.,0.,0.,0.,0.,0.]#, np.zeros(16, dtype=float), np.zeros(16, dtype=float)]
        gW = [0.] * len(learning_rates)
        gW_bw = [0.] * len(learning_rates)

        for index in range(int(n_batches_train)):

            # CHANGE THE INDEX OF THE MINI BATCH (= CLAMP X AND INITIALIZE THE HIDDEN AND OUTPUT LAYERS WITH THE PERSISTENT PARTICLES)
            net.change_mini_batch_index(index)

            # FREE PHASE
            net.free_phase_fw(n_it_neg, epsilon)
            net.free_phase_bw(n_it_neg_bw, epsilon)

            # MEASURE THE ENERGY, COST AND ERROR AT THE END OF THE FREE PHASE RELAXATION
            measures = net.measure()
            measures_sum = [measure_sum + measure for measure_sum,measure in zip(measures_sum,measures)]
            measures_avg = [measure_sum / (index+1) for measure_sum in measures_sum]
            measures_avg[2] *= 100. # FW error rate as a percentage
            measures_avg[5] *= 100. # BW error rate as a percentage
            
            stdout.write("\r%2i-train-%5i FW: E=%.1f C=%.5f error=%.3f%% | BW: E=%.1f C=%.5f error=%.3f%%" % (epoch, (index+1)*batch_size, measures_avg[0], measures_avg[1], measures_avg[2], measures_avg[3], measures_avg[4], measures_avg[5]))
            #stdout.write("\r" % (measures_avg[6], measures_avg[7]))
            stdout.flush()

            # WEAKLY CLAMPED PHASE

            # Choose the signs of alpha and beta at random
            sign = 2*np.random.randint(0,2)-1 # random sign +1 or -1
            beta = np.float32(sign*beta) 
            sign = 2*np.random.randint(0,2)-1 # random sign +1 or -1
            alpha = np.float32(sign*alpha)

            Delta_logW = net.weakly_clamped_phase_fw(n_it_pos, epsilon, beta, *learning_rates)
            gW = [gW1 + Delta_logW1 for gW1,Delta_logW1 in zip(gW,Delta_logW)]

            Delta_logW_bw = net.weakly_clamped_phase_bw(n_it_pos_bw, epsilon, alpha, *learning_rates)
            gW_bw = [gW1 + Delta_logW1 for gW1,Delta_logW1 in zip(gW_bw,Delta_logW_bw)]

        stdout.write("\n")
        #dlogW = [100. * gW1 / n_batches_train for gW1 in gW]
        dlogW_bw = [100. * gW1 / n_batches_train for gW1 in gW_bw]
        #print("   FW:"+" ".join(["dlogW%i=%.3f%%" % (k+1,dlogW1) for k,dlogW1 in enumerate(dlogW)]))
        print("   BW:"+" ".join(["dlogW%i=%.3f%%" % (k+1,dlogW1) for k,dlogW1 in enumerate(dlogW_bw)]))
        net.training_curves["fw training error"].append(measures_avg[2])
        net.training_curves["bw training error"].append(measures_avg[5])

        ### VALIDATION ###
        
        # CUMULATIVE SUM OF VALIDATION ENERGY, VALIDATION COST AND VALIDATION ERROR
        measures_sum = [0.,0.,0.,0.,0.,0.]

        for index in range(int(n_batches_valid)):

            # CHANGE THE INDEX OF THE MINI BATCH (= CLAMP X AND INITIALIZE THE HIDDEN AND OUTPUT LAYERS WITH THE PERSISTENT PARTICLES)
            net.change_mini_batch_index(n_batches_train+index)

            # FREE PHASE
            net.free_phase_fw(n_it_neg, epsilon)
            net.free_phase_bw(n_it_neg_bw, epsilon)   
            
            # MEASURE THE ENERGY, COST AND ERROR AT THE END OF THE FREE PHASE RELAXATION
            measures = net.measure()
            measures_sum = [measure_sum + measure for measure_sum,measure in zip(measures_sum,measures)]
            measures_avg = [measure_sum / (index+1) for measure_sum in measures_sum]
            measures_avg[2] *= 100. # FW error rate as a percentage
            measures_avg[5] *= 100. # BW error rate as a percentage
            stdout.write("\r%2i-valid-%5i FW: E=%.1f C=%.5f error=%.3f%% | BW: E=%.1f C=%.5f error=%.3f%%" % (epoch, (index+1)*batch_size, measures_avg[0], measures_avg[1], measures_avg[2], measures_avg[3], measures_avg[4], measures_avg[5]))
            stdout.flush()

        stdout.write("\n")

        net.training_curves["fw validation error"].append(measures_avg[2])
        net.training_curves["bw validation error"].append(measures_avg[5])

        duration = (time.clock() - start_time) / 60.
        print("   duration=%.1f min" % (duration))


        # SAVE THE PARAMETERS OF THE NETWORK AT THE END OF THE EPOCH
        #net.save_params(epoch=epoch)
    # SAVE THE NETWORK AT THE END OF ALL EPOCHS
    
    print("FW TRAINING ERROR")
    print(net.training_curves["fw training error"])

    print("FW VALID ERROR")
    print(net.training_curves["fw validation error"])

    print("BW TRAINING ERROR")
    print(net.training_curves["bw training error"])

    print("BW VALID ERROR")
    print(net.training_curves["bw validation error"])
    net.save_params(epoch=epoch)
    f = open(net.path.replace('.save', 'RBEP_SPLIT_BIDIR.errors'), 'wb')

    to_dump = net.training_curves["fw training error"], net.training_curves["fw validation error"], net.training_curves["bw training error"], net.training_curves["bw validation error"]
    pickle.dump(to_dump, f, protocol=2) # For Python 2 compatability
    f.close()


# HYPERPARAMETERS FOR A NETWORK WITH 1 HIDDEN LAYER
net1 = "net1", {
"hidden_sizes" : [50],
"n_epochs"     : 20,
"batch_size"   : 20,
"n_it_neg"     : 25,
"n_it_pos"     : 5,
"n_it_neg_bw"     : 25, # Keep over 20?
"n_it_pos_bw"     : 5	,
"epsilon"      : np.float32(.5),
"alpha"         : np.float32(.5),
"beta"         : np.float32(.5),
"learning_rates"       : [np.float32(.1), np.float32(.05)]#[np.float32(.1), np.float32(.05)]
}

# HYPERPARAMETERS FOR A NETWORK WITH 2 HIDDEN LAYERS
net2 = "net2", {
"hidden_sizes" : [100,100],
"n_epochs"     : 60,
"batch_size"   : 20,
"n_it_neg"     : 150,
"n_it_pos"     : 6,
"n_it_neg_bw"     : 150, # Keep over 20?
"n_it_pos_bw"     : 6	,
"epsilon"      : np.float32(.5),
"alpha"         : np.float32(.5),
"beta"         : np.float32(1.),
"learning_rates"       : [np.float32(.4), np.float32(.1), np.float32(.1)]#[np.float32(.4), np.float32(.1), np.float32(.01)]
}

# HYPERPARAMETERS FOR A NETWORK WITH 3 HIDDEN LAYERS
net3 = "net3", {
"hidden_sizes" : [100,100,100],
"n_epochs"     : 500,
"batch_size"   : 20,
"n_it_neg"     : 500,
"n_it_pos"     : 8,
"n_it_neg_bw"     : 500,
"n_it_pos_bw"     : 8,
"epsilon"      : np.float32(.5),
"alpha"         : np.float32(.5),
"beta"         : np.float32(1.),
"learning_rates"  : [np.float32(.128), np.float32(.032), np.float32(.008), np.float32(.002)]
}



if __name__ == "__main__":

    # TRAIN A NETWORK WITH 1 HIDDEN LAYER
    #train_net(Network(*net1))
    train_net(Network(*net2))
