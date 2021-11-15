import numpy as np
import random
import pandas as pd

class Game(object):

    def __init__(self, df):
        self.df = df
        self.reset()
        
    def _update_state(self, action):
        
        '''Here we update our state'''
        self.curr_idx += 1
        self.curr_time = self.df.index[self.curr_idx]
        self.curr_price = self.df['Close'][self.curr_idx]
        self.curr_volume = self.df['Volume'][self.curr_idx]
        self.typical_price = (self.df['High'][self.curr_idx]+self.df['Low'][self.curr_idx]+self.df['Close'][self.curr_idx])/3
        #self.close_high = foo((self.df['High'][self.curr_idx]-self.df['Close'][self.curr_idx]), (self.df['High'][self.curr_idx]-self.df['Low'][self.curr_idx]))
        self.pseudo_vol = foo((self.df['High'][self.curr_idx]-self.df['Low'][self.curr_idx]), self.df['Open'][self.curr_idx])
        self.pnl = (-self.entry + self.curr_price)*self.position

        _k = list(map(float,str(self.curr_time.time()).split(':')[:2]))

        
        '''This is where we define our policy and update our position'''

        if action == 0:  # Hold
            if self.position == 0:
                self.entry = self.curr_price
                self.start_idx = self.curr_idx
            else:
                self.total_pnl = self.balance + self.entry*self.position + self.pnl
                self.reward = self.pnl
                
        
        elif action == 2: # Buy
            if self.position == -1:

                # one short round finished
                self.is_over = True
                self._get_reward() 
                self.total_pnl += self.pnl
                self.balance = self.total_pnl

                # new round
                self.position = 0  
                self.entry = self.curr_price
                self.start_idx = self.curr_idx
                
   
            elif self.position == 0:
                self.position = 1
                self.entry = self.curr_price
                self.start_idx = self.curr_idx
                self.is_over = False
                self.balance -= self.curr_price
                #self.total_pnl = self.balance + self.curr_price
            else: 
                self.total_pnl = self.balance + self.entry*self.position + self.pnl
                self.reward = self.pnl
            
        elif action == 1: # Sell
            if self.position == 1:
                # one long round finished
                self.is_over = True
                self._get_reward() 
                self.total_pnl += self.pnl
                self.balance = self.total_pnl

                # new round
                self.position = 0  
                self.entry = self.curr_price
                self.start_idx = self.curr_idx

            elif self.position == 0:
                self.position = -1
                self.entry = self.curr_price
                self.start_idx = self.curr_idx
                self.is_over = False
                self.balance += self.curr_price
            else:
                self.total_pnl = self.balance + self.entry*self.position + self.pnl
                self.reward = self.pnl

        self._assemble_state()

    def _assemble_state(self):
        self.state = np.array([])
        self.state = np.append(self.state,self.curr_price)
        self.state = np.append(self.state, self.typical_price)
        #self.state = np.append(self.state, self.close_high)
        self.state = np.append(self.state, self.pseudo_vol)
        self.state = np.append(self.state,self.curr_volume)
        self.state = np.append(self.state,self.position)
        #self.state = np.append(self.state,self._time_of_day)
        #self.state = np.append(self.state,self._day_of_week)
        self.state = (np.array(self.state)-np.mean(self.state))/np.std(self.state)
        

    def _get_reward(self):
        if self.position == 1 and self.is_over:
            pnl = (self.curr_price - self.entry)
            self.reward = pnl#-(self.curr_idx - self.start_idx)/1000.
        elif self.position == -1 and self.is_over:
            pnl = (-self.curr_price + self.entry)
            self.reward = pnl#-(self.curr_idx - self.start_idx)/1000.
        #print('entry:',self.entry,'exit:',self.curr_price,'pos:',self.position,'pnl:',pnl,self.reward)
        return self.reward
            
    def observe(self):
        return np.array([self.state])

    def act(self, action):
        self._update_state(action)
        reward = self.reward
        game_over = self.is_over
        return self.observe(), reward, game_over

    def reset(self):
        self.is_over = False
        self.pnl = 0
        self.total_pnl = 0
        self.balance = 0
        self.curr_idx = 0
        self.start_idx = self.curr_idx
        self.curr_time = self.df.index[self.curr_idx]
        self.curr_volume = self.df['Volume'][self.curr_idx]
        self.curr_price = self.df['Close'][self.curr_idx]
        self.entry = self.curr_price
        self.state = []
        self.position = 0
        self.reward = 0
        self.typical_price = (self.df['High'][self.curr_idx]+self.df['Low'][self.curr_idx]+self.df['Close'][self.curr_idx])/3
        self.close_high = (self.df['High'][self.curr_idx]-self.df['Close'][self.curr_idx])/(self.df['High'][self.curr_idx]-self.df['Low'][self.curr_idx])
        self.pseudo_vol = (self.df['High'][self.curr_idx]-self.df['Low'][self.curr_idx])/self.df['Open'][self.curr_idx]
        self._update_state(0)

def foo(x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return 1