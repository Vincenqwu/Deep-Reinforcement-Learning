from env import Game
from A2C_Agent import Agent
import tensorflow as tf
import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
import traceback
from IPython import display
from datetime import datetime, timedelta, time, date
import matplotlib.dates as mdates



def Train(df,epoch,chkpt_dir, alpha = 0.0005):
    # parameters
    num_actions = 3 
    env = Game(df)
    agent = Agent(n_actions=num_actions, alpha = alpha, chkpt_dir = chkpt_dir)

    # Train

    pnls = []
    for e in range(epoch):
        loss = 0.
        env.reset()
        state = env.observe()
        for t in range(len(df)-2):
            # get next action
            state = env.observe()
            state = tf.convert_to_tensor(state)
            action = agent.choose_action(state)
            # apply action, get rewards and new state
            new_state, reward, game_over = env.act(action)
            agent.learn(state, reward, tf.convert_to_tensor(new_state)) 

        prt_str = ("Epoch {:03d} | pnl {:.2f}".format(e, env.total_pnl))

        print(prt_str)
        pnls.append(env.total_pnl)

    agent.save_models()
    fig =plt.figure(figsize = (8, 5))
    plt.plot(range(len(pnls)),pnls)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('P&L', fontsize=10)
    print("Average P&L is: ",sum(pnls)/len(pnls))   
    return 


def Test(df, ticker, alpha, chkpt_dir):
    env = Game(df)
    agent = Agent(alpha = alpha, chkpt_dir = chkpt_dir)
    agent.load_models()
    env.reset()
    state = env.observe()
    pnls_record = [0]
    action_record = [0]
    for t in range(len(df)-2):
        # get next action
        state = env.observe()
        state = tf.convert_to_tensor(state)
        action = agent.choose_action(state)
        # apply action, get rewards and new state
        print('#', t, 
            'action', action, 
            'pos:' , env.position, 
            'reward:', "{:.3f}".format(env.reward), 
            'balance:', "{:.3f}".format(env.balance), 
            'pnl:', "{:.3f}".format(env.total_pnl))
        if (env.position == 1 and action == 2) or (env.position == -1 and action ==1):
            action_record.append(0)
        else:
            action_record.append(action)
        pnls_record.append(env.total_pnl)
        new_state, reward, game_over = env.act(action)
        agent.learn(state, reward, tf.convert_to_tensor(new_state))
    action_record.append(0)
    pnls_record.append(pnls_record[-1])
    action_record = np.array(action_record)
    df['buy_mask'] = (action_record == 2)
    df['sell_mask'] = (action_record == 1)
    df['pnls'] = pnls_record
    plot_result(df[:-2], ticker)
    print(env.total_pnl)


def plot_result(df, ticker):
    times = pd.date_range(df.index[0], df.index[-1],len(df))


    
    fig =plt.figure(figsize = (10,12))
    fig.add_subplot(211)
    plt.plot(times, df['Close'].values, linewidth=1, alpha = 0.8, label = ticker)

    
    #print(len(times[df['buy_mask']))
    #print(df['Close'][df['buy_mask'][:-2]].values[:-2])
    plt.scatter(times[df['buy_mask']], df['Close'][df['buy_mask']].values, label='Buy',
                    marker='*', alpha = 0.8, color='r',s = 25)
    plt.scatter(times[df['sell_mask']], df['Close'][df['sell_mask']].values, label='Sell',
                    marker='*', alpha = 0.8, color='g',s = 25)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    plt.legend()
    
    fig.add_subplot(212)
    plt.plot(times, df['pnls'].values,label = 'P&L')
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('P&L', fontsize=10)

    #xfmt = mdates.DateFormatter('%m-%d-%y %H:%M')
    #fig.axes[0].xaxis.set_major_formatter(xfmt) 
    plt.legend()
    plt.show()
    
    