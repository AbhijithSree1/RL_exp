import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import time
# Note: The 'core' import might be different in your version, adjust if needed.
# e.g., import spinup.algos.pytorch.vpg.core as core
import spinup.algos.pytorch.opg.core as core
from spinup.utils.logx import EpochLogger


class OPGBuffer:
    """
    A buffer for storing trajectories experienced by a OPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size    # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, rew=self.rew_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
    
    def reset(self):
        self.ptr, self.path_start_idx = 0, 0


def opg(env_fn, actor_critic_observer=core.MLPActorCriticObserver, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, max_imagine_steps=1000, epochs=50, obs_epochs = 5, imagine_epochs = 5, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, obs_lr = 7e-4, rew_lr = 7e-4, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):

    """
    Proximal Policy Gradient (OPG) with Observer
    This function implements the OPG algorithm with an observer model for environment dynamics.
    Ideas behind this algorithm is that humans after an experience in an environment, build a mental model of how the
    environment works and use that model to imagine and plan their next actions. Similarly, here we train an observer model to predict the next state
    given the current state and action, and a reward learner to predict the reward from the next state. After some initial epochs of normal environment interaction and
    training, we use the observer and reward learner to imagine trajectories and update the policy based on those imagined trajectories.
    """


    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic_observer(env.observation_space, env.action_space, **ac_kwargs)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v, ac.observer])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t obs: %d\n'%var_counts)

    # Set up experience buffer
    buf = OPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    obs_optimizer = Adam(ac.observer.parameters(), lr=obs_lr)
    rlearn_optimizer = Adam(ac.rlearner.parameters(), lr=rew_lr)
    loss_fn = torch.nn.MSELoss(reduction="mean")

    logger.setup_pytorch_saver(ac)

    def update(data):

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            pi_optimizer.step()
        
        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))
        
    def TrainObserver(obs_epochs):
        data = buf.get()
        obs = data['obs'][:-1]
        act = data['act'][:-1]
        next_obs = data['obs'][1:]
        rew = data['rew'][:1]
        epoch = 0
        average_loss_obs = 0
        average_loss_rlearn = 0
        while epoch < obs_epochs:   
            epoch += 1  
            obs_optimizer.zero_grad()
            pi,_ = ac.observer(obs, act)
            # Negative log likelihood
            loss_obs = -(pi.log_prob(next_obs).sum(-1)).mean()
            loss_obs.backward()
            obs_optimizer.step()
            average_loss_obs += loss_obs.item()
            average_loss_obs /= epoch

            rlearn_optimizer.zero_grad()
            pred_rew = ac.rlearner(next_obs)
            loss_rlearn = loss_fn(pred_rew, rew)
            loss_rlearn.backward()
            rlearn_optimizer.step()
            average_loss_rlearn += loss_rlearn.item()
            average_loss_rlearn /= epoch

            print('Observer Epoch: ', epoch, ' Loss: ', average_loss_obs, ' Reward Learner Loss: ', average_loss_rlearn)
        logger.store(ObsAvgLoss=average_loss_obs)
        logger.store(RLearnAvgLoss=average_loss_rlearn)

    def imagine_step():
        data = buf.get()
        buf_imagine = OPGBuffer(obs_dim, act_dim, max_imagine_steps, gamma, lam)
        imagine_epoch = 0
        while imagine_epoch < imagine_epochs:
            imagine_epoch += 1
            index = np.random.randint(0, data['obs'].shape[0]-1)
            
            obs, act = data['obs'][index], data['act'][index]   # take random initial sample from buffer

            with torch.no_grad():
                for i in range(max_imagine_steps):                           # imagine for 1000 steps
                    # print(i)
                    pi,_ = ac.observer(obs.unsqueeze(0), act.unsqueeze(0))    # predict next state prop dist from current state and action
                    obs_next = pi.sample().squeeze(0)                           # sample next state
                    rew = ac.rlearner(obs_next.unsqueeze(0)).squeeze()                      # predict reward from next state
                    a, v, logp = ac.step(obs_next)                           # get action from policy    
                    # print(obs_next, act, rew, v, logp)
                    buf_imagine.store(obs_next, act, rew, v, logp)
                    obs = obs_next
                    act = torch.as_tensor(a, dtype=torch.float32)

            buf_imagine.finish_path(v)
            data_imagine = buf_imagine.get()
            update(data_imagine)
            buf_imagine.reset()

    start_time = time.time()
    (o, _), ep_ret, ep_len = env.reset(), 0, 0
    obs_warmup_epochs = 2

    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, terminated, truncated, _ = env.step(a)
            d = terminated or truncated
            ep_ret += r
            ep_len += 1

            buf.store(o, a, r, v, logp)
            logger.store(VVals=v)
            
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                (o, _), ep_ret, ep_len = env.reset(), 0, 0

        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        TrainObserver(obs_epochs)
        data = buf.get()
        update(data)  
        if epoch > obs_warmup_epochs:
            print('Imagining steps and updating policy...')
            imagine_step()
        buf.reset()
        

        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ObsAvgLoss', average_only=True)
        logger.log_tabular('RLearnAvgLoss', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v3')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1) # Set default CPU to 1
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='opg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.env, args.seed)

    opg(lambda : gym.make(args.env), actor_critic_observer=core.MLPActorCriticObserver,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)