import collections
import random
from typing import List, Tuple, Dict, Any, Callable, Tuple
import math
from collections import defaultdict
import numpy as np
from copy import deepcopy
import pysnooper
import itertools
import pandas as pd

## helper function to work on 2D tuple
## originally work on Numpy array, but it is not hashable ... 
## Then I think directly work on 2d tuple may be better....

def slice_tuple(rowstart, rowend, colstart, colend, tuple_input):
    # just slicing tuple like numpy array[rowstart: rowend, colstart: colend]
    list_1 = []
    for i in range(rowstart, rowend):
        list_2 = []
        for j in range(colstart, colend):
            ele = tuple_input[i][j]
            list_2.append(ele)
        list_1.append(tuple(list_2))
    newtuple = tuple(list_1)
    return newtuple

def max_2d(tuple_2d):
    # max value in 2d tuple
    return max([max(x) for x in tuple_2d])


def sum_2d(tuple_2d):
    # sum of value in 2d tuple
    return sum([sum(x) for x in tuple_2d])

def set_tuple(rowstart, rowend, colstart, colend, old_tuple, value):
    # replace value in tuple slicing
    list_1 = [list(x) for x in old_tuple]
    for i in range(rowstart, rowend):
        for j in range(colstart, colend):
            list_1[i][j] = value

    return tuple(tuple(x) for x in list_1)

def find_left(atuple):
    # find left edge of box stack block within a row
    list_of_index =[]
    for i in range(len(atuple)):
        if atuple[i] == 1:
            list_of_index.append(i)
    return min(list_of_index)

def find_right(atuple):
     # find right edge of box stack block within a row
    list_of_index =[]
    for i in range(len(atuple)):
        if atuple[i] == 1:
            list_of_index.append(i)
    return max(list_of_index)
        
    

# An abstract class representing a Markov Decision Process (MDP).
class MDP:
    # Return the start state.
    def startState(self) -> Tuple: raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(
        self, state: Tuple) -> List[Any]: raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(
        self, state: Tuple, action: Any) -> List[Tuple]: raise NotImplementedError("Override me")

    def discount(self): raise NotImplementedError("Override me")

    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self):
        raise NotImplementedError("Override me")

        



class BoxMDP(MDP):
    #                          
    def __init__(self, boxList, space_width, space_height):

        # box is (height, width)!!!!!!!!!!!
        self.boxList = tuple(boxList)
        self.space_width = space_width
        self.space_height = space_height
        self.space = tuple([tuple([0 for i in range(self.space_width)]) for j in range(self.space_height)])

    # Return the start state.
    # first element to record the space, second element is box list, third element is box picked

    def startState(self):
        return (self.space, self.boxList, None)

    def actions(self, state) :
        # if box is not picked: action is like 'pick0' , means we pick #0 box in the box tuple
        if len(state[1]) >0 and state[2] is None:
            actions = ['pick'+str(i) for i in range(len(state[1]))]
        # if box is picked: action is like 'drop0', means we drop the box on hand to location 0 (x-axis)
        elif state[2] is not None:
            avail_loc = list(range(self.space_width - state[2][1] + 1))
            actions = ['drop'+str(loc) for loc in avail_loc]
        # if no box on hand, no box left in tuple, we have to end
        elif len(state[1]) == 0 and state[2] is None:
            actions = ['end']
        return actions

    def isEnd(self, state):
        ## if fallen, or the list is depleted & no box on hand, then it is end
        if (state[0] is None) or (len(state[1]) == 0 and state[2] is None ):
            return True
        return False

    ## occupy the space
    def drop_box(self, space, box, position):

        space = deepcopy(space)

        # drop box
        fall = False
        index_y = 0
        index_x = position
        
        try:
            maxdd = max_2d(slice_tuple(index_y, index_y + box[0] , index_x, index_x+ box[1], space)  ) 
        except:
            print(index_y, index_y + box[0] , index_x, index_x+ box[1], space)
            
        # if space has been occupied then we add index by 1
        while max_2d(slice_tuple(index_y, index_y + box[0] , index_x, index_x+ box[1], space)  ) >0:
            index_y = index_y +1

        space = set_tuple(index_y,index_y + box[0] , index_x, index_x+ box[1] , space, 1)

        ## fall if gravity center go out
        ### get the y below the new box
        if index_y >= 1:
            #if not on ground, calculate the gravity center
            below_index_y = index_y -1
            center_of_gravity = index_x + (box[1])/2
            # check whether two side of gravity center both have suport 
            if sum_2d( slice_tuple(below_index_y, below_index_y+1,  index_x , math.ceil(center_of_gravity), space) ) == 0 or \
                sum_2d( slice_tuple(below_index_y, below_index_y+1, math.ceil(center_of_gravity)-1, index_x + (box[1]) ,space) ) == 0:
            # if fall set space as empty
                fall = True
        
        return space, fall


    def calculate_gap(self, space):
        # gaps on horizontal space
        y_index = 0
        total_hor_gap = 0
        ## while y is small than row number
        # print(space)
        while y_index < len(space):
            row_gap = 0

            if sum(space[y_index]) >0:
            # find the leftmost block and right most block
                left = find_left(space[y_index])
                right = find_right( space[y_index])
                
                row_gap = (right -left+1) - sum(space[y_index])
                # print(left, right, row_gap)
            y_index += 1
            total_hor_gap += row_gap
        return total_hor_gap

    def calculate_unstability(self, space):
        space_array = np.asarray(space)
        # gaps on horizontal space
        x_index = 0
        total_vert_gap = 0

        if space is None :
            print('end state already')
            return 

        while x_index < len(space_array[0]):
            col_gap = 0
            if sum(space_array[:,x_index]) >0:

                top = 0
                bottom = max(np.where(space_array[:,x_index] == 1)[0])
                col_gap = bottom -top - (sum(space_array[:,x_index])-1)

            x_index += 1
            total_vert_gap += col_gap
        return total_vert_gap

    def reward(self, space):
        '''
        This reward function is very arbituary, subject to change

        5*box_count -(self.calculate_unstability(space) + self.calculate_gap(space)) 

        '''
        box_count = len(self.boxList)
        # arbitually using 5* box count minus sum of unstability and gap
#         return 5*box_count -(self.calculate_unstability(space) + self.calculate_gap(space))
        return -(self.calculate_unstability(space) + self.calculate_gap(space))


    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state: Tuple, action: str) -> List[Tuple]:
        
        if self.isEnd(state):
            return []

        state_list_to_return = []

        if 'pick' in action and len(state[1]) >0:
            ## randomly pick a box, prob is 1/len
            prob = 1

            ## add box ,newbox list to new state. Space does not change. Reward is 0
            # for box in state[1]:

            box_list = list(state[1])

            box_index = int(action[-1])
            box = box_list[box_index]

            _ = box_list.remove(box)
            new_box_tuple = tuple(box_list)

            state_list_to_return.append( ((state[0], new_box_tuple, box), prob, 0))


        if 'drop' in action and state[2] is not None:  # state[2] is box
            
            # we can drop to any x-position (use left of box as the marker of box position)
            prob = 1

            position = int(action[-1])

            # for position in list(range(0, self.space_width - state[2][1] + 1)):
                
            # drop box state[0] is space; state[2] is box
            new_space, fall = self.drop_box(state[0], state[2], position)

            if fall is True:
                # if fall, set the space to None
                state_list_to_return.append(
                    ( (None, state[1], None), prob, -999))
            # if there is still box left to drop,
            elif len(state[1]) > 0:
                state_list_to_return.append(
                    ((new_space, state[1], None), prob, 0))
            # if there is not fall and no box left ,calculate the reward
            else:
                return [((new_space, state[1], None), 1, self.reward(new_space))]
        
        if 'end' in action:

            return [((new_space, (), None), 1, self.reward(new_space))]


        return state_list_to_return

        # END_YOUR_CODE

    def discount(self):
        return 1

    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.

    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)
        print("%d states" % len(self.states))
#         print(self.states)




############################################################

# Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.  The simulator (see
# simulate()) will call getAction() to get an action, perform the action, and
# then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
# its parameters.
class RLAlgorithm:
    # Your algorithm will be asked to produce an action given a state.
    def getAction(
        self, state: Tuple) -> Any: raise NotImplementedError("Override me")

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |newState|.
    def incorporateFeedback(self, state: Tuple, action: Any, reward: int,
                            newState: Tuple): raise NotImplementedError("Override me")



class QLearningAlgorithm(RLAlgorithm):
    def __init__(self, actions: Callable, discount: float, featureExtractor: Callable, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state: Tuple, action: Any) -> float:
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state: Tuple) -> Any:
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            # print('second route of getting action')
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state: Tuple, action: Any, reward: int, newState: Tuple) -> None:
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        Q_opt_s_a_prime = 0.0

        if newState is None:
            return

        else:
            # print(f'newState is {newState}')

            # print(f'(self.getQ(newState, action), action) is {[ (self.getQ(newState, action), action) for action in self.actions(newState) ]}')

            step = self.getStepSize()
                
            Q_opt_s_a_prime = max((self.getQ(newState, action), action)
                                  for action in self.actions(newState))[0]

            for item in self.featureExtractor(state, action):
                key, value = item

                self.weights[key] -= step*(self.getQ(key, action) -
                                           (reward + self.discount*Q_opt_s_a_prime))*value

        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.


def identityFeatureExtractor(state: Tuple, action: Any) -> List[Tuple[Tuple, int]]:
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]
        

def NewFeatureExtractor(state: Tuple, action: str) -> List[tuple]:
    space, box_tuple, box_on_hand = state

    # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
    feature_list = []

    feature_list.append(
        (
            ('total', total, action),
            1)
    )
    
    space 

    if counts is not None:
        # print('bitmask', tuple([1 if counts[i] != 0 else 0 for i in range(len(counts)) ]))
        feature_tuple = (('bitmask',
                          tuple(
                              [1 if counts[i] != 0 else 0 for i in range(len(counts))]),
                          action), 1)
        feature_list.append(feature_tuple)

    if counts is not None:
        for i in range(len(counts)):
            feature_tuple = ((i, counts[i], action), 1)
            feature_list.append(feature_tuple)

    return feature_list


# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.


def simulate(mdp: MDP, rl: RLAlgorithm, numTrials=10, maxIterations=1000, verbose=False,
             sort=False):
    # Return i in [0, ..., len(probs)-1] with probability probs[i].
    def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target:
                return i
        raise Exception("Invalid probs: %s" % probs)

    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
        state = mdp.startState()
        sequence = [state]
        totalDiscount = 1
        totalReward = 0
        # loop through iteration
        for _ in range(maxIterations):
            # get one action
            action = rl.getAction(state)
            # return transitions -- (new state,prob, reward)
            transitions = mdp.succAndProbReward(state, action)
            # print(f'transition table {transitions}')
            if sort:
                transitions = sorted(transitions)

            # if no tranition available, break
            if len(transitions) == 0:
                # print('calling from rl.incorporateFeedback(state, action, 0, None)')
                rl.incorporateFeedback(state, action, 0, None)
                break

            # Choose a random transition
            i = sample([prob for newState, prob, reward in transitions])

            # print(f'transitions[i] is {transitions[i]}')

            newState, prob, reward = transitions[i]

            # print(f'newState {newState} prob {prob}  reward {reward}')


            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)

            # incorcoprate feedback (looks like updating some instance property)
            # print('calling from rl.incorporateFeedback(state, action, reward, newState)')
            rl.incorporateFeedback(state, action, reward, newState)
            totalReward += totalDiscount * reward
            totalDiscount *= mdp.discount()
            state = newState
        if verbose:
            print(("Trial %d (totalReward = %s): %s" %
                   (trial, totalReward, sequence)))
        totalRewards.append(totalReward)
    return totalRewards



def simulate_QL_over_MDP(mdp,featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches. Remember to
    # set your explorationProb to zero after simulate.
    # BEGIN_YOUR_CODE

    rl = QLearningAlgorithm(mdp.actions, mdp.discount(),
                            featureExtractor,
                            0.2)
    mdp.computeStates()

    totalRewards = simulate(mdp, rl, numTrials=30000,
                      maxIterations=2000, verbose=False, sort=False)

    rl.explorationProb = 0

#     for state in mdp.states:
#         print(f'State: {state} ; Learned RL policy: {rl.getAction(state)}')

    # print(len(mdp.states))
    return totalRewards

class Baseline():
    def __init__(self, MDP_instance ,k):
        # if just initiation, then we pick k boxes to be seen
        self.MDP_instance = MDP_instance
        self.space, self.box_tuple, _ = MDP_instance.startState()
        box_list = list(self.box_tuple)
        random.shuffle(box_list)
        self.seen_boxes = box_list[:k]
        self.unseen_boxes = box_list[k:]
    
#     @pysnooper.snoop()
    def box_grab(self):

        ## find widest boxed in the seen list
        box_widest_width = max(self.seen_boxes, key = lambda x: x[1])[1]

        seen_box_copy = deepcopy(self.seen_boxes)
        
        widest_boxes =[]

        while max(seen_box_copy, key = lambda x: x[1])[1] == box_widest_width:

            wide_index = seen_box_copy.index(max(seen_box_copy, key = lambda x: x[1]))

            widest_boxes.append(seen_box_copy.pop(wide_index))

            if len(seen_box_copy) == 0:
                break

        ## find tallest boxed in the widest box list
        the_box = max(widest_boxes, key = lambda x: x[0])

        # return tuple without the box
        idx = self.seen_boxes.index(the_box)

        self.seen_boxes = self.seen_boxes[:idx] + self.seen_boxes[idx+1:]
        
        #if still unseen box, random pick one more box from the unseen boxes
        if len(self.unseen_boxes) > 0:

            # random pick one more box from the unseen boxes
            random.shuffle(self.unseen_boxes)

            selected_box = self.unseen_boxes.pop(0)

            self.seen_boxes.append(selected_box)
        
        return the_box
    
    
    def find_position_to_drop_baseline(self, space, box):
        '''
        ignored many edge cases. 
        only suitable for baseline case (wide to narrow, tall to short)

        '''
        y_index = 0
         #   if not suppassing the world's heigh
        while y_index < len(space):
            if y_index + box[0] > len(space):
                return None

            # check wether there is empty space on the y_axis
            if min(space[y_index]) == 0:
                #find where the empty space begins
                empty_space_starting = space[y_index].index(min(space[y_index]))
                # if space is enough
                if len(space[0]) - (empty_space_starting ) >= box[1]:
                    position = empty_space_starting
                    return position
            y_index += 1
    
#     @pysnooper.snoop()
    def drop_box_base(self, space, box, position):

        space = deepcopy(space)

        # drop box
        fall = False
        index_y = 0
        index_x = position

        # if space has been occupied then we add index by 1
        while max_2d(slice_tuple(index_y, index_y + box[0] , index_x, index_x+ box[1], space)  ) >0:
            index_y = index_y +1

        space = set_tuple(index_y,index_y + box[0] , index_x, index_x+ box[1] , space, 1)

        ## fall if gravity center go out
        ### get the y below the new box
        if index_y >= 1:
            #if not on ground, calculate the gravity center
            below_index_y = index_y -1
            center_of_gravity = index_x + (box[1])/2
            # check whether two side of gravity center both have suport 
            if sum_2d( slice_tuple(below_index_y, below_index_y+1,  index_x , math.ceil(center_of_gravity), space) ) == 0 or \
                sum_2d( slice_tuple(below_index_y, below_index_y+1, math.ceil(center_of_gravity)-1, index_x + (box[1]) ,space) ) == 0:
            # if fall set space as empty
                fall = True

        return space, fall


    def run_baseline(self):
        '''
        run baseline strategy and return the rewards
        '''
        
        ### see 5 box
        box_number = len(self.box_tuple) 
        box_count = 1

        while box_count <= box_number:
#             print('seen_boxes:', self.seen_boxes)
#             print('unseen_boxes',self.unseen_boxes)

            the_box = self.box_grab()

#             print(the_box)

            drop_position = self.find_position_to_drop_baseline(self.space, the_box)

            # if drop_position is string, that means we got
            if drop_position is None:
                raise BaseException('out of world bound')

            self.space, fall = self.drop_box_base(self.space, the_box, drop_position)
            
        #        if fall, return 0 reward
            if fall is True:
                return -999, self.space

            box_count += 1 

#             print('seen_boxes:',self.seen_boxes)

#             print('unseen_boxes:',self.unseen_boxes)

#             print('**********************************************')

        return self.MDP_instance.reward(self.space), self.space



def generate_n_random_box(n, biggest_size):
    
    def random_box(biggest_size):
        return (random.choice(list(range(1, biggest_size+1))), random.choice(list(range(1, biggest_size+1))))

    random_boxes = []
    for i in range(n):
        random_boxes.append(random_box(biggest_size))
    return random_boxes

#############


box_numbers = [3, 4, 5]
world_sizes = [3, 4, 5]

reward_dict = collections.defaultdict(list)

for i in itertools.product(box_numbers, world_sizes):
    box_number = i[0]
    world_size = i[1]
    print(f'box number {box_number}, world size {world_size}')
    
    run_time = 1
    
    baseline_rewards = []
    rl_rewards = []

    while run_time <= 20:
        #                                        # size width height
        randomMDP = BoxMDP(generate_n_random_box(box_number, 3), world_size, 3*box_number)

        baseline_instance = Baseline(randomMDP, 3)
        b_reward, _ = baseline_instance.run_baseline()

        baseline_rewards.append(b_reward)

#         print('finish baseline')

        r_reward = simulate_QL_over_MDP(randomMDP, identityFeatureExtractor)

        mean_reward = round(sum(r_reward)/len(r_reward), 1)

        rl_rewards.append(mean_reward)

#         print('finish rl')

        run_time += 1
    
    for x, y in zip(baseline_rewards, rl_rewards):
#         print(f'reward comparison: baseline: {x} RL: {y}') 
        reward_dict[f'baseline_n_{box_number}_w_{world_size}'].append(x)
        reward_dict[f'RL_n_{box_number}_w_{world_size}'].append(y)
        

raw_csv = pd.DataFrame(reward_dict)
raw_csv.to_csv('raw_results_reward_1_small.csv')

mean_csv = pd.DataFrame(raw_csv.apply(lambda x: round(np.mean(x),4), axis = 0), columns=['mean'])
mean_csv.to_csv('mean_results_reward_1_small.csv')