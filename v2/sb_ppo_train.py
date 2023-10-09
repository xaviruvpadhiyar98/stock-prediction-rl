from pathlib import Path
from utils import *

def main():
    MODEL = "PPO"
    FRAMEWORK = "sb"
    LEARN_DESCRIBE = "best_param_early_stopping"
    SEED = 1337
    NUM_ENVS = 512

    train_envs, trade_env = get_train_trade_environment(
        framework="sb", num_envs=NUM_ENVS, seed=SEED
    )

    trained_model = get_best_ppo_model(train_envs, SEED)
    
    multiplier = 500
    total_timesteps = NUM_ENVS * trained_model.n_steps * multiplier

    model_filename = Path(TRAINED_MODEL_DIR) / f"{MODEL}.zip"
    tb_log_name = (
        f"{MODEL}_{FRAMEWORK}_" 
        f"{LEARN_DESCRIBE}_{NUM_ENVS}_" 
        f"{trained_model.n_steps}_{SEED}"
    )

    try:
        trained_model.learn(
            total_timesteps=total_timesteps,
            callback=TensorboardCallback(eval_env=trade_env, model_filename=model_filename, seed=SEED),
            tb_log_name=tb_log_name,
            log_interval=1,
            progress_bar=True,
            reset_num_timesteps=True,
        )

    except KeyboardInterrupt:
        trained_model.save(model_filename)

    trained_model.save(model_filename)    
    train_envs.close()
    trade_env.close()


if __name__ == "__main__":
    main()
