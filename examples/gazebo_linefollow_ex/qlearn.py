import random
import pickle


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        try:
            with open(f"{filename}.pickle", 'rb') as file:
                self.q = pickle.load(file)
            print("Loaded file: {}".format(filename + ".pickle"))
        except FileNotFoundError:
            print("File not found:", filename + ".pickle")


    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
         # Assuming self.Q is the dictionary or list of arrays you want to save
        with open(f"{filename}.pickle", 'wb') as file:
            pickle.dump(self.q, file)
        
        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        action = None
        q = None
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
            q = self.getQ(state, action)
        else:
            max_q = 0
            best_actions = []
            for action in self.actions:
                current_q = self.getQ(state, action)
                if  current_q > max_q:
                    best_actions = [action]
                    max_q = current_q
                elif current_q == max_q:
                    best_actions.append(action)

            # If no best action found (empty list), pick a random action
            if not best_actions:
                action = random.choice(self.actions)
                q = self.getQ(state, action)
            else:
                action = random.choice(best_actions)
                q = max_q

        return(action, q) if return_q else action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        old_q = self.getQ(state1, action1)
        max_q_next = max([self.getQ(state2, a) for a in self.actions])
        new_q = old_q + self.alpha * (reward + self.gamma * max_q_next - old_q)
        self.q[(state1, action1)] = new_q
