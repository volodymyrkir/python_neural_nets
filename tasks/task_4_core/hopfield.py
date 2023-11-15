import numpy as np

class Hopfield:
    def __init__(self, shape):
        self.neurons_count = shape[0] * shape[1]

    def train_weights(self, train_data, learning_rule='hebb'):
        self.train_data = np.array(train_data)
        examples_count = len(train_data)
        
        # initialize weights
        W = np.zeros((self.neurons_count, self.neurons_count))
        rho = np.sum([np.sum(t) for t in train_data]) / (examples_count * self.neurons_count)

        if learning_rule == 'hebb':
            # Hebbian rule
            for t in self.train_data:
                t = t - rho            
                W += np.outer(t, t)

            # Make diagonal element of W into 0
            W /= examples_count
            np.fill_diagonal(W, 0)

            self.W = W
        elif learning_rule == 'storkey':
            # Storkey rule
            for t in self.train_data:
                W += np.outer(t, t) / self.neurons_count
                net = np.dot(W, t)

                pre = np.outer(t, net)
                post = np.outer(net, t)

                W -= np.add(pre, post) / self.neurons_count

            np.fill_diagonal(W, 0)

            self.W = W
        elif learning_rule == 'demircigil':
            pass
        else:
            raise Exception("Not supported update function") 
         
    
    def predict(self, data, iterations=20, threshold=0, energy_function='classic', update_function='sync', async_iterations=100):
        # Copy to avoid call by reference 
        copied_data = np.copy(data)
        
        # Define predict list
        predicted = []
        for i in range(len(data)):
            predicted.append(self.update(copied_data[i], iterations, threshold, energy_function, update_function, async_iterations))
        return predicted
    
    def update_async(self, initial_state, iterations, threshold, energy_closure, async_iterations):
        """
        Asynchronous update
        """
        # Compute initial state energy
        state = initial_state
        e = energy_closure(state)
        
        # Iteration
        for i in range(iterations):
            for j in range(async_iterations):
                # Select random neuron
                index = np.random.randint(0, self.neurons_count) 
                # Update state
                state[index] = np.sign(self.W[index].T @ state - threshold)
            
            # Compute new state energy
            e_new = energy_closure(state)
            
            # state is converged
            if e == e_new:
                return state
            # Update energy
            e = e_new
        return state
    
    def update_sync(self, initial_state, iterations, threshold, energy_closure):
        """
        Synchronous update
        """
        # Compute initial state energy
        state = initial_state

        e = energy_closure(state)
        
        # Iteration
        for i in range(iterations):
            # Update state
            state = np.sign(self.W @ state - threshold)
            # Compute new state energy
            e_new = energy_closure(state)
            
            # state is converged
            if e == e_new:
                return state
            # Update energy
            e = e_new
        return state
    
    def update_demircigil(self, initial_state, iterations, threshold, energy_closure):
        """
        Demircigil asynchronous update
        """
        # Compute initial state energy
        state = initial_state
        e = energy_closure(state)
        
        def update_neuron(l):
            # Get state with selected neuron flipped
            state_flipped_pos = np.copy(state)
            state_flipped_pos[l] = 1.0

            state_flipped_neg = np.copy(state)
            state_flipped_neg[l] = -1.0

            # Update state
            return np.sign(-energy_closure(state_flipped_pos) + energy_closure(state_flipped_neg))
        next = np.vectorize(update_neuron)

        # Iteration
        for i in range(iterations):
            state = next(np.array(range(self.neurons_count)))

            # Compute new state energy
            e_new = energy_closure(state)
            
            # state is converged
            if e == e_new:
                return state
            # Update energy
            e = e_new
        return state

    def update(self, initial_state, iterations, threshold, energy_function, update_function, async_iterations):
        def energy_closure(state):
            return self.energy(state, threshold, energy_function)
        
        if update_function == 'async':
            return self.update_async(initial_state, iterations, threshold, energy_closure, async_iterations)
        elif update_function == 'sync':
            return self.update_sync(initial_state, iterations, threshold, energy_closure)
        elif update_function == 'demircigil':
            return self.update_demircigil(initial_state, iterations, threshold, energy_closure)
        else:
            raise Exception("Not supported update function") 
    
    def energy_classic(self, state, threshold):
        return -0.5 * state @ self.W @ state + np.sum(state * threshold)

    def energy_exponential(self, state):
        return -np.sum(np.exp((self.train_data @ state) / 10))

    def energy(self, state, threshold, energy_function):
        if energy_function == "classic":
            return self.energy_classic(state, threshold)
        elif energy_function == "exp":
            return self.energy_exponential(state)
