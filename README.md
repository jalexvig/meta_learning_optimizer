# Meta-Learning an Optimizer with RL

See [this explanation](http://jalexvig.github.io/meta-learning-optimizer/) for more details/motivation.

### Example use

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Options

```
usage: main.py [-h] [--env ENV] [--model_dir MODEL_DIR] [--seed SEED]
               [--n_iter N_ITER] [--dont_normalize_advantages]
               [--mix_halflife MIX_HALFLIFE] [--mix_start MIX_START]
               [--batch_size BATCH_SIZE] [--discount DISCOUNT]
               [--ep_len EP_LEN] [--num_lstm_units NUM_LSTM_UNITS]
               [--grad_reg GRAD_REG] [--no_xover] [--render]
               [--histogram_parameters] [--reset] [--debug]
               [--run_name RUN_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             Name of environment.
  --model_dir MODEL_DIR
                        Directory to save run information in.
  --seed SEED           Random seed.
  --n_iter N_ITER       Number iterations.
  --dont_normalize_advantages, -dna
                        Advantage normalization can help with stability. Turns
                        that feature off.
  --mix_halflife MIX_HALFLIFE
                        Halflife of gradient/action mixture.
  --mix_start MIX_START
                        Starting weight for magnitude of gradient
                        contributions.
  --batch_size BATCH_SIZE
                        Batch size when collecting gradients for policy.
  --discount DISCOUNT   Discount rate for rewards.
  --ep_len EP_LEN       Number of steps before performing an update.
  --num_lstm_units NUM_LSTM_UNITS
                        Number LSTM units in policy hidden layer.
  --grad_reg GRAD_REG   Cap for l2 norm of gradients.
  --no_xover            Treat each gradient as separate input to policy.
  --render              Not implemented.
  --histogram_parameters
                        Record histogram of parameters.
  --reset               Delete the existing model directory and start training
                        from scratch.
  --debug               Run session in a CLI debug session.
  --run_name RUN_NAME   Name of run.
```
